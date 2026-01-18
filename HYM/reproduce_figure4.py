import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from pymatgen.core import Structure, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mp_api.client import MPRester
import matplotlib.pyplot as plt
import networkx as nx
from defect_prediction_model import CGT
import sys

# ==============================================================================
# 1. Configuration & Utilities
# ==============================================================================

API_KEY = "lM8Zb4HVPItyYB9SYj5qSReTQNi0zcDH"
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
MP_PDOS_DATA = os.path.join(WORK_DIR, "mp_pdos_data.json")
ORBITAL_ELECTRONS = os.path.join(WORK_DIR, "orbital_electrons.json")
OUTPUT_PLOT = os.path.join(WORK_DIR, "reproduced_figure4.png")

print("Starting reproduce_figure4.py", flush=True)

# Load orbital electrons data
if not os.path.exists(ORBITAL_ELECTRONS):
    print(f"Error: {ORBITAL_ELECTRONS} not found.", flush=True)
    sys.exit(1)

with open(ORBITAL_ELECTRONS, 'r') as f:
    ORBITAL_EMBEDDINGS = json.load(f)

# Precompute orbital counts
ORBITAL_COUNTS_MAP = {}
for element, info in ORBITAL_EMBEDDINGS.items():
    s_count = info.get('1s', 0) + info.get('2s', 0) + info.get('3s', 0) + info.get('4s', 0) + info.get('5s', 0) + info.get('6s', 0) + info.get('7s', 0)
    p_count = info.get('2p', 0) + info.get('3p', 0) + info.get('4p', 0) + info.get('5p', 0) + info.get('6p', 0) + info.get('7p', 0)
    d_count = info.get('3d', 0) + info.get('4d', 0) + info.get('5d', 0) + info.get('6d', 0)
    f_count = info.get('4f', 0) + info.get('5f', 0) + info.get('6f', 0)
    ORBITAL_COUNTS_MAP[element] = [s_count, p_count, d_count, f_count]

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

def structure_to_graph(structure, radius=2.5):
    G = nx.Graph()
    for i, site in enumerate(structure):
        G.add_node(i, element=site.species_string, coords=site.coords)
    
    cutoff = radius
    # Get all neighbors within cutoff
    all_neighbors = structure.get_all_neighbors(cutoff)
    
    for i, neighbors in enumerate(all_neighbors):
        for neighbor in neighbors:
            # neighbor is (site, distance, index, image)
            # pymatgen 2024+ might change neighbor format, usually it's object or tuple
            # older versions: (neighbor_site, distance, index, image)
            # newer versions: Neighbor object
            if hasattr(neighbor, 'index'):
                j = neighbor.index
                dist = neighbor.nn_distance
            else:
                 # Fallback for tuple
                j = neighbor[2]
                dist = neighbor[1]
                
            if i != j: # Avoid self loops if any
                 G.add_edge(i, j, weight=dist)
                 
    return G

def process_structure(structure, real_pdos, energies):
    """
    Convert pymatgen Structure and real PDOS to PyG Data object
    """
    sga = SpacegroupAnalyzer(structure)
    space_group_number = sga.get_space_group_number()
    
    G = structure_to_graph(structure, radius=2.5)
    
    edge_index = []
    edge_attr = []
    atom_fea = []
    orbital_counts = []
    
    for i, site in enumerate(structure):
        element = site.species_string
        # Handle cases like "Lu" -> "Lu"
        # If species is ElementComposition, take the first one
        
        orbital_counts.append(ORBITAL_COUNTS_MAP.get(element, [0,0,0,0]))
        
        # Atomic number (0-based index for embedding?)
        # Model uses Embedding(118), so indices 0-117.
        # Original code uses atomic_number directly (1-based).
        # We match original code: use atomic number directly.
        el = Element(element)
        atom_fea.append(el.number) 

    # Edges
    for u, v, data in G.edges(data=True):
        edge_index.append([u, v])
        edge_index.append([v, u])
        edge_attr.append(data['weight'])
        edge_attr.append(data['weight'])
        
    if len(edge_index) == 0:
        # Handle isolated atoms case (should not happen in crystal)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = np.array(edge_attr)
        
    # Gaussian expansion
    gdf = GaussianDistance(dmin=0, dmax=2.5, step=0.2)
    if len(edge_attr) > 0:
        edge_attr = gdf.expand(edge_attr)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_attr = torch.zeros((0, 14), dtype=torch.float) # 14 features for default settings

    # Targets
    # real_pdos: dictionary {'s': [], 'p': [], 'd': []}
    # Model output is (4, 201) corresponding to s, p, d, f
    # We need to stack them.
    # mp_pdos_data.json only has s, p, d. We pad f with zeros.
    
    s_dos = np.array(real_pdos['orbitals']['s'])
    p_dos = np.array(real_pdos['orbitals']['p'])
    d_dos = np.array(real_pdos['orbitals']['d'])
    
    # Interpolate to 201 points if needed, but assuming input is already consistent or we resize
    # The model expects fixed input/output size (201).
    # We should interpolate energies and DOS to 201 points within [-10, 10] eV range.
    
    target_energies = np.linspace(-10, 10, 201)
    input_energies = np.array(real_pdos['energies'])
    
    # Interpolate
    s_dos_interp = np.interp(target_energies, input_energies, s_dos, left=0, right=0)
    p_dos_interp = np.interp(target_energies, input_energies, p_dos, left=0, right=0)
    d_dos_interp = np.interp(target_energies, input_energies, d_dos, left=0, right=0)
    f_dos_interp = np.zeros_like(target_energies) # Dummy f
    
    pdos_target = np.stack([s_dos_interp, p_dos_interp, d_dos_interp, f_dos_interp], axis=0)
    
    # Create Data object
    data = Data(
        x=torch.tensor(atom_fea, dtype=torch.long),
        edge_index=edge_index,
        edge_attr=edge_attr,
        energies=torch.tensor(target_energies, dtype=torch.float).unsqueeze(0), # [1, 201]
        orbital_counts=torch.tensor(orbital_counts, dtype=torch.float),
        y=torch.tensor(pdos_target, dtype=torch.float).unsqueeze(0), # [1, 4, 201]
        structure_name=real_pdos.get('formula', 'Unknown')
    )
    
    return data

# ==============================================================================
# 2. Main Logic
# ==============================================================================

def main():
    # 1. Load MP PDOS Data
    print(f"Loading data from {MP_PDOS_DATA}", flush=True)
    if not os.path.exists(MP_PDOS_DATA):
        print(f"Error: {MP_PDOS_DATA} not found. Please run fetch_and_plot_real_pdos.py first.", flush=True)
        return
        
    with open(MP_PDOS_DATA, 'r') as f:
        mp_data = json.load(f)
        
    formulas = list(mp_data.keys())
    print(f"Loaded data for: {formulas}", flush=True)
    
    # 2. Fetch Structures
    print("Fetching structures from Materials Project...", flush=True)
    structures = {}
    try:
        with MPRester(API_KEY) as mpr:
            for formula in formulas:
                print(f"Searching for {formula}...", flush=True)
                try:
                    # Search for material ID
                    docs = mpr.materials.summary.search(formula=formula, fields=["material_id", "is_stable", "structure"])
                    if not docs:
                        print(f"Warning: No structure found for {formula}", flush=True)
                        continue
                    
                    # Pick best doc
                    best_doc = docs[0]
                    for doc in docs:
                        if doc.is_stable:
                            best_doc = doc
                            break
                            
                    structures[formula] = best_doc.structure
                    print(f"Got structure for {formula}", flush=True)
                except Exception as e:
                    print(f"Error fetching structure for {formula}: {e}", flush=True)
    except Exception as e:
        print(f"MPRester error: {e}", flush=True)

    # 3. Prepare Dataset
    print("Preparing dataset...", flush=True)
    dataset = []
    for formula, struct in structures.items():
        if formula in mp_data:
            try:
                data = process_structure(struct, mp_data[formula], mp_data[formula]['energies'])
                data.formula = formula
                dataset.append(data)
            except Exception as e:
                print(f"Error processing {formula}: {e}", flush=True)
            
    if not dataset:
        print("No valid data to train on.", flush=True)
        return

    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 4. Initialize Model
    print("Initializing model...", flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    
    try:
        model = CGT(edge_dim=14, out_dim=201*4).to(device)
    except Exception as e:
        print(f"Model init error: {e}", flush=True)
        return

    # Check for pre-trained weights
    weights_path = os.path.join(WORK_DIR, "GNNs-PDOS", "shujujione", "JPCA2025", "CGT_phys_weights.pth")
    if os.path.exists(weights_path):
        print(f"Loading pre-trained weights from {weights_path}...", flush=True)
        try:
            # Original model might have been saved as state_dict or full model
            # Usually state_dict
            checkpoint = torch.load(weights_path, map_location=device)
            # Check if checkpoint is a dict or state_dict
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                 model.load_state_dict(checkpoint['state_dict'])
            elif isinstance(checkpoint, dict):
                 # Try loading directly if keys match
                 # Filter keys if necessary (e.g. 'module.' prefix from DataParallel)
                 new_state_dict = {}
                 for k, v in checkpoint.items():
                     if k.startswith('module.'):
                         new_state_dict[k[7:]] = v
                     else:
                         new_state_dict[k] = v
                 model.load_state_dict(new_state_dict, strict=False)
            else:
                 print("Unknown checkpoint format.", flush=True)
            
            print("Pre-trained weights loaded successfully!", flush=True)
            do_training = False
        except Exception as e:
            print(f"Error loading weights: {e}. Will train from scratch.", flush=True)
            do_training = True
    else:
        print("Pre-trained weights not found. Will train from scratch.", flush=True)
        do_training = True
        
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 5. Train (if needed)
    if do_training:
        print("Training model...", flush=True)
        model.train()
        epochs = 100 # Quick training
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # Forward
                try:
                    out = model(batch)
                    
                    # Loss
                    loss = criterion(out, batch.y.to(device))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                except Exception as e:
                    print(f"Training error at epoch {epoch}: {e}", flush=True)
                    break
                
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}", flush=True)
    else:
        print("Skipping training (using pre-trained weights).", flush=True)

    # 6. Predict & Plot
    print("Generating predictions...", flush=True)
    model.eval()
    
    results = {}
    
    with torch.no_grad():
        for data in dataset:
            data_gpu = data.to(device)
            # data.batch = torch.zeros(data.x.size(0), dtype=torch.long).to(device) # Single graph batch
            # Actually DataLoader handles batching, but for single item we need to set batch
            # if we pass it manually. But let's just reuse the data object which works if we use a loader of size 1
            # or manually add batch attribute.
            
            # Easier: pass through model
            # The model expects 'batch' attribute.
            data_gpu.batch = torch.zeros(data_gpu.x.size(0), dtype=torch.long, device=device)
            try:
                out = model(data_gpu) # [1, 4, 201]
                pred_pdos = out.cpu().numpy()[0] # [4, 201]
                
                results[data.formula] = {
                    "energies": data.energies.numpy()[0],
                    "real": data.y.numpy()[0],
                    "pred": pred_pdos
                }
            except Exception as e:
                print(f"Prediction error for {data.formula}: {e}", flush=True)

    # 7. Visualization
    print("Plotting...", flush=True)
    try:
        plot_results(results)
    except Exception as e:
        print(f"Plotting error: {e}", flush=True)

def plot_results(results):
    n_samples = len(results)
    if n_samples == 0: return
    
    cols = 3
    rows = (n_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    if n_samples == 1: axes = [axes]
    else: axes = axes.flatten()
    
    orbitals = ['s', 'p', 'd'] # We ignore f for plotting as it's usually empty/small
    colors = {'s': '#1f77b4', 'p': '#ff7f0e', 'd': '#2ca02c'}
    
    for i, (formula, res) in enumerate(results.items()):
        if i >= len(axes): break
        ax = axes[i]
        
        energies = res["energies"]
        real_dos = res["real"]
        pred_dos = res["pred"]
        
        # Clip predicted DOS to be non-negative
        pred_dos = np.maximum(pred_dos, 0)
        
        offset = 0
        for idx, orb in enumerate(orbitals):
            # Real
            r_d = real_dos[idx]
            # Pred
            p_d = pred_dos[idx]
            
            # Fill Real
            ax.fill_between(energies, r_d + offset, offset, color=colors[orb], alpha=0.3, label=f"{orb} (Real)" if i==0 else "")
            ax.plot(energies, r_d + offset, color=colors[orb], linewidth=1, alpha=0.8)
            
            # Plot Pred (Dashed)
            ax.plot(energies, p_d + offset, color=colors[orb], linestyle='--', linewidth=2, label=f"{orb} (Pred)" if i==0 else "")
            
            offset += (np.max(r_d) if len(r_d)>0 else 1) * 0.5 + 1
            
        ax.set_title(f"{formula}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Energy (eV) - Fermi Level")
        if i % cols == 0:
            ax.set_ylabel("Density of States")
        ax.set_xlim(-10, 10)
        ax.grid(True, linestyle=':', alpha=0.5)
        
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=6)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Figure saved to {OUTPUT_PLOT}", flush=True)

if __name__ == "__main__":
    main()
