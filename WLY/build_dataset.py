import os
import json
import torch
import numpy as np
from mp_api.client import MPRester
from pymatgen.electronic_structure.dos import CompleteDos
from pymatgen.core.structure import Structure
from torch_geometric.data import Data
from torch import nn
from typing import Optional
from scipy.interpolate import interp1d
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. RBF Class (Copied for self-containment)
# ==========================================
class RBFExpansion(nn.Module):
    def __init__(self, vmin=0, vmax=8, bins=40, lengthscale=None):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer("centers", torch.linspace(self.vmin, self.vmax, self.bins))
        if lengthscale is None:
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale
        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (distance.unsqueeze(1) - self.centers) ** 2)

# ==========================================
# 2. Helper Functions
# ==========================================
def to_list(obj):
    if hasattr(obj, 'tolist'): return obj.tolist()
    if isinstance(obj, list): return obj
    return list(obj)

def get_orbital_counts(structure, orbital_db):
    orbital_counts = []
    orbital_map = ['s', 'p', 'd', 'f']
    for site in structure:
        elem_sym = site.specie.symbol
        counts = [0, 0, 0, 0]
        if elem_sym in orbital_db:
            orb_data = orbital_db[elem_sym]
            for orb_key, count in orb_data.items():
                orb_type = orb_key[-1]
                if orb_type in orbital_map:
                    counts[orbital_map.index(orb_type)] += count
        orbital_counts.append(counts)
    return torch.tensor(orbital_counts, dtype=torch.float)

def process_dos_data(dos, target_grid_size=201):
    # Prepare Grid
    grid_min, grid_max = -10.0, 10.0
    target_energies = np.linspace(grid_min, grid_max, target_grid_size)
    
    # Raw Energies
    raw_energies = np.array(dos.energies)
    e_fermi = dos.efermi
    shifted_energies = raw_energies - e_fermi
    
    # Aggregate PDOS
    orbital_types = ['s', 'p', 'd', 'f']
    aggregated_pdos = {orb: np.zeros_like(raw_energies) for orb in orbital_types}
    
    if hasattr(dos, 'pdos'):
        for site, orbital_data in dos.pdos.items():
            for orb_name, spin_data in orbital_data.items():
                # Handle Orbital Enum or String
                orb_str = str(orb_name)
                # If it's an enum like Orbital.s, str() might give "Orbital.s" or "s" depending on version
                # Usually name attribute works if it's an Enum
                if hasattr(orb_name, 'name'):
                    orb_str = orb_name.name
                
                orb_type = orb_str[0].lower()
                if orb_type not in orbital_types: continue
                
                dens_total = np.zeros_like(raw_energies)
                for spin, density in spin_data.items():
                     dens_total += np.array(density)
                aggregated_pdos[orb_type] += dens_total
                
    # Interpolate
    final_pdos_matrix = np.zeros((4, target_grid_size))
    for i, orb in enumerate(orbital_types):
        f_interp = interp1d(shifted_energies, aggregated_pdos[orb], kind='linear', bounds_error=False, fill_value=0.0)
        final_pdos_matrix[i, :] = f_interp(target_energies)
        
    return torch.from_numpy(final_pdos_matrix).float(), torch.from_numpy(target_energies).float()

# ==========================================
# 3. Main Builder Class
# ==========================================
class DatasetBuilder:
    def __init__(self, api_key, output_dir="WLY/dataset"):
        self.api_key = api_key
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load orbital DB
        with open("WLY/orbital_electrons.json", 'r') as f:
            self.orbital_db = json.load(f)
            
    def build(self, limit=20):
        print(f"Searching for materials (target: {limit} samples)...")
        with MPRester(self.api_key) as mpr:
            # Search for stable materials
            # We fetch more candidates than limit because some might not have DOS
            candidate_limit = limit * 5
            # Ensure we don't ask for too many at once (API limit ~1000)
            chunk_size = min(candidate_limit, 500) 
            
            docs = mpr.summary.search(
                is_stable=True, 
                fields=["material_id", "structure", "formula_pretty"],
                num_chunks=int(np.ceil(candidate_limit / chunk_size)),
                chunk_size=chunk_size
            )
            
            # Since chunk_size might not be strict limit in wrapper, slice it
            docs = docs[:candidate_limit]
            print(f"Found {len(docs)} candidates. Starting download & processing...")
            
            success_count = 0
            for i, doc in enumerate(docs):
                if success_count >= limit:
                    break
                    
                mp_id = doc.material_id
                formula = doc.formula_pretty
                
                try:
                    print(f"[{i+1}/{len(docs)}] Processing {mp_id} ({formula})...")
                    
                    # 1. Get DOS
                    # We need to fetch DOS separately as it's heavy
                    dos_data = mpr.get_dos_by_material_id(mp_id)
                    if not dos_data:
                        print(f"  - No DOS data returned for {mp_id}")
                        continue
                        
                    # 2. Process DOS
                    pdos_tensor, energies_tensor = process_dos_data(dos_data)
                    
                    # 3. Process Structure (Graph)
                    structure = doc.structure
                    
                    # Node Features
                    atomic_numbers = [site.specie.number for site in structure]
                    x = torch.tensor(atomic_numbers, dtype=torch.long).unsqueeze(1)
                    
                    # Edge Features
                    cutoff = 8.0
                    all_neighbors = structure.get_all_neighbors(r=cutoff, include_index=True)
                    edge_src, edge_dst, edge_dist = [], [], []
                    for src_idx, neighbors in enumerate(all_neighbors):
                        for neighbor in neighbors:
                            edge_src.append(src_idx)
                            edge_dst.append(neighbor.index)
                            edge_dist.append(neighbor.nn_distance)
                            
                    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
                    edge_dist_tensor = torch.tensor(edge_dist, dtype=torch.float)
                    
                    rbf = RBFExpansion(vmin=0, vmax=8.0, bins=14)
                    edge_attr = rbf(edge_dist_tensor)
                    
                    # Orbital Counts
                    orbital_counts = get_orbital_counts(structure, self.orbital_db)
                    
                    # 4. Save Data
                    # We save everything in one Data object for simplicity in training
                    data = Data(
                        x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=pdos_tensor, # Label: (4, 201)
                        energies=energies_tensor.unsqueeze(0),
                        orbital_counts=orbital_counts,
                        mp_id=mp_id,
                        formula=formula
                    )
                    
                    torch.save(data, os.path.join(self.output_dir, f"{mp_id}.pt"))
                    success_count += 1
                    print(f"  - Saved successfully.")
                    
                except Exception as e:
                    print(f"  - Failed: {str(e)}")
                    
            print(f"Dataset build complete. {success_count}/{len(docs)} saved to {self.output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    
    builder = DatasetBuilder(args.key)
    builder.build(limit=args.limit)
