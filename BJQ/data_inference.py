import os
import json
import sqlite3
import torch
import numpy as np
from ase.io import read
from ase.db import connect
from pymatgen.io.ase import AseAtomsAdaptor
from torch_geometric.data import Data, DataLoader
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import Element

# 引入项目中的 data.py 中的部分工具类，或者重新实现
from scipy.ndimage import gaussian_filter1d

def smooth(array, sigma=3):
    return gaussian_filter1d(array, sigma=sigma)

class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 / self.var ** 2)

class Imp2DDataset:
    def __init__(self, db_path, orbital_electrons_path, radius=2.5, dmin=0, step=0.2):
        self.db_path = db_path
        self.orbital_electrons_path = orbital_electrons_path
        self.radius = radius
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        
        # Load orbital electrons configuration
        with open(self.orbital_electrons_path) as f:
            self.emb = json.load(f)
            
        self.emb2 = {element: [
                    info.get('1s', 0), 
                    info.get('2s', 0), info.get('2p', 0), 
                    info.get('3s', 0), info.get('3p', 0), info.get('3d', 0), 
                    info.get('4s', 0), info.get('4p', 0), info.get('4d', 0), info.get('4f', 0),
                    info.get('5s', 0), info.get('5p', 0), info.get('5d', 0), info.get('5f', 0),
                    info.get('6s', 0), info.get('6p', 0), info.get('6d', 0), info.get('6f', 0),
                    info.get('7s', 0), info.get('7p', 0)
                ] for element, info in self.emb.items()}

        self.orbital_counts = {}
        for element, info in self.emb.items():
            s_count = sum([info.get(f'{n}s', 0) for n in range(1, 8)])
            p_count = sum([info.get(f'{n}p', 0) for n in range(2, 8)])
            d_count = sum([info.get(f'{n}d', 0) for n in range(3, 7)])
            f_count = sum([info.get(f'{n}f', 0) for n in range(4, 7)])
            self.orbital_counts[element] = [s_count, p_count, d_count, f_count]

    def get_defect_info(self, row_data):
        """
        Extract defect information from DB row metadata.
        
        Note: The current imp2d.db seems to contain 'defecttype' in key_value_pairs.
        We will use this to determine vacancy_type.
        
        For 'vacancy_dists', we need the defect position. 
        However, the DB might not explicitly store the defect coordinate if it's implicitly defined by the structure.
        Or 'site' might give a hint.
        
        For this implementation, we will try to infer or use placeholders if specific coordinates are missing.
        Based on the previous `data.py` logic, we need to return:
        - has_defect (bool)
        - defect_coords (list [x,y,z])
        - defect_type (int)
        """
        kv_pairs = row_data.key_value_pairs
        
        defect_type_str = kv_pairs.get('defecttype', 'none')
        has_defect = defect_type_str != 'none'
        
        # Map defect string types to integers
        defect_map = {
            'none': 0,
            'adsorbate': 1,
            'interstitial': 2,
            'substitution': 3,
            'vacancy': 4
        }
        defect_type_int = defect_map.get(defect_type_str, 0)
        
        # Determine defect coordinates
        # Since exact defect coordinates might require complex parsing of the supercell generation history,
        # we will use a simplified approach:
        # If it's an adsorbate or interstitial, the 'dopant' atom is likely the center.
        # If it's a vacancy, the missing atom is the center (harder to find from just the defect structure).
        
        defect_coords = [0.0, 0.0, 0.0] # Default center
        
        # Try to find the dopant atom for ads/int/sub
        if has_defect and defect_type_int in [1, 2, 3]:
            dopant_species = kv_pairs.get('dopant')
            # Convert ASE atoms to Pymatgen structure to find the dopant
            atoms = row_data.toatoms()
            # This is a heuristic: find the atom that matches the dopant species
            # If there are multiple, this might pick the first one.
            # For supercells with one defect, this is usually fine.
            for atom in atoms:
                if atom.symbol == dopant_species:
                    defect_coords = atom.position
                    break
        
        return has_defect, defect_coords, defect_type_int

    def process_row(self, row):
        # Filter: Only process converged calculations with valid formation energy
        # Note: key_value_pairs is a dict in ASE row object (accessed via row.key_value_pairs)
        kv = row.key_value_pairs
        if not kv.get('converged', False):
            return None
            
        eform = kv.get('eform')
        if eform is None or np.isnan(eform):
            return None

        # Convert ASE Atoms to Pymatgen Structure
        atoms = row.toatoms()
        structure = AseAtomsAdaptor.get_structure(atoms)
        
        # Get Space Group
        try:
            sga = SpacegroupAnalyzer(structure)
            space_group_number = sga.get_space_group_number()
        except:
            space_group_number = 1 # Fallback

        # --- Graph Construction ---
        # Nodes
        atom_fea = []
        elec_conf = []
        orbital_counts_list = []
        
        for site in structure:
            element = site.species_string
            # Handle cases where element might not be in our JSON (e.g. noble gases if not present)
            # Assuming standard elements are present.
            if element not in self.emb2:
                 # Fallback or error
                 print(f"Warning: Element {element} not in orbital_electrons.json. Skipping row.")
                 return None

            elec_conf.append(self.emb2[element])
            orbital_counts_list.append(self.orbital_counts[element])
            atom_fea.append([Element(element).number])

        # Edges
        # Using Pymatgen's get_all_neighbors for efficiency or existing logic
        # Here implementing simple distance check similar to original data.py
        
        # Optimization: Use neighbor list
        neighbors = structure.get_all_neighbors(self.radius, include_index=True)
        
        edge_index = []
        edge_attr = []
        
        for i, neighbor_list in enumerate(neighbors):
            for neighbor in neighbor_list:
                # neighbor is (site, distance, index)
                dist = neighbor[1]
                j = neighbor[2]
                
                edge_index.append([i, j])
                edge_attr.append(dist)
        
        if len(edge_index) == 0:
             # Handle isolated atoms case if necessary
             edge_index = np.empty((2, 0), dtype=np.int64)
             edge_attr = np.empty((0), dtype=np.float32)
        else:
             edge_index = np.array(edge_index).T
             edge_attr = np.array(edge_attr)
             edge_attr = self.gdf.expand(edge_attr)

        # --- Defect Features ---
        has_defect, defect_coords, defect_type = self.get_defect_info(row)
        
        if has_defect:
            atom_coords = torch.tensor(structure.cart_coords, dtype=torch.float32)
            d_coords = torch.tensor(defect_coords, dtype=torch.float32).unsqueeze(0)
            vacancy_dists = torch.norm(atom_coords - d_coords, dim=1)
        else:
            vacancy_dists = torch.ones(len(structure), dtype=torch.float32) * 100.0
            
        # --- Targets (PDOS) ---
        # The DB might NOT contain the full PDOS grid as expected by the model (4x201).
        # We need to check if 'data' column or specific keys contain PDOS.
        # Based on user input, we are doing INFERENCE ("进行预测"), so we might not have ground truth PDOS.
        # Or if we are training, we need it.
        # User said: "利用这几个数据集合，使用刚刚训练好的模型，进行预测，执行训练"
        # It seems contradictory: "predict" AND "execute training".
        # Assuming we need to TRAIN on this data if it has labels, or PREDICT if it doesn't.
        # Let's check the DB for 'pdos' or similar data.
        # For now, we will create dummy targets for prediction mode, 
        # or try to extract if available.
        
        # Placeholder targets
        pdos = torch.zeros((1, 4, 201)) 
        energies = torch.zeros((1, 201))
        
        # Try to extract ground truth if available in DB
        # eform: Formation energy per atom usually? Or total? 
        # imp2d.db 'eform' usually means formation energy.
        # Band gap: usually not explicitly in key_value_pairs unless calculated.
        # p_band_center: not in DB.
        
        formation_energy_val = row.get('eform', 0.0)
        if formation_energy_val is None or np.isnan(formation_energy_val):
             formation_energy_val = 0.0
             
        # Check if band gap is available (sometimes in 'gap' or 'bandgap')
        band_gap_val = row.get('gap', 0.0)
        if band_gap_val is None:
             band_gap_val = row.get('bandgap', 0.0)
        if band_gap_val is None or np.isnan(band_gap_val):
             band_gap_val = 0.0
             
        p_band_center = torch.tensor([0.0])
        formation_energy = torch.tensor([formation_energy_val])
        band_gap = torch.tensor([band_gap_val]) 

        # Tensors
        x = torch.tensor(atom_fea, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        space_group_number = torch.tensor([space_group_number]).unsqueeze(0)
        elec_conf = torch.tensor(elec_conf, dtype=torch.float)
        orbital_counts = torch.tensor(orbital_counts_list, dtype=torch.float)
        vacancy_type = torch.tensor([defect_type], dtype=torch.long)
        
        data = Data(
            mp_id=row.get('name', f'id_{row.id}'),
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            energies=energies,
            space_group_number=space_group_number,
            p_band_center=p_band_center,
            y=pdos,
            elec_conf=elec_conf,
            orbital_counts=orbital_counts,
            formation_energy=formation_energy,
            band_gap=band_gap,
            vacancy_dists=vacancy_dists,
            vacancy_type=vacancy_type
        )
        
        return data

    def load_all(self, limit=None):
        data_list = []
        with connect(self.db_path) as db:
            rows = db.select(limit=limit)
            for row in rows:
                data = self.process_row(row)
                if data:
                    data_list.append(data)
        return data_list

if __name__ == '__main__':
    # Test loading
    db_path = r'd:\Github hanjia\whuphy-attention\BJQ\imp2d.db'
    orb_path = r'd:\Github hanjia\whuphy-attention\BJQ\orbital_electrons.json'
    dataset = Imp2DDataset(db_path, orb_path)
    data = dataset.load_all(limit=5)
    print(f"Loaded {len(data)} samples.")
    print(data[0])
