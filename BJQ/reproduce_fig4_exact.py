
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
import json

# Add GNNs-PDOS to path to import models
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
GNN_PDOS_PATH = os.path.join(WORK_DIR, "GNNs-PDOS")
sys.path.append(GNN_PDOS_PATH)

# Import the specific model class user requested
try:
    from models.CGT_phys import CGT
    print("Successfully imported CGT from models.CGT_phys")
except ImportError as e:
    print(f"Import failed: {e}")
    # Fallback or exit? Let's try to adjust path if needed
    sys.exit(1)

# Configuration
DATASET_DIR = os.path.join(GNN_PDOS_PATH, "shujujione", "JPCA2025", "dataset")
WEIGHTS_PATH = os.path.join(GNN_PDOS_PATH, "shujujione", "JPCA2025", "CGT_phys_weights.pth")
OUTPUT_PLOT = os.path.join(WORK_DIR, "reproduced_figure4_exact.png")

# Target materials from Figure 4 (based on previous knowledge)
TARGET_IDS = {
    "mp-1189682": "LuB2C",
    "mp-1220094": "NdSmFe17N3",
    "mp-1215264": "ZrScVNi3",
    "mp-1187277": "Tb2Y2O5",
    "mp-1184402": "Eu3Dy",
    "mp-754261": "La4FeO8"
}

def smooth(array, sigma=3):
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(array, sigma=sigma)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Initialize Model
    # Note: CGT_phys.py CGT class init: (edge_dim=14, out_dim=201*4, seed=123)
    model = CGT(edge_dim=14, out_dim=201*4).to(device)

    # 2. Load Pre-trained Weights
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Weights file not found: {WEIGHTS_PATH}")
        return

    print(f"Loading weights from {WEIGHTS_PATH}...")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    
    # Handle state dict keys (remove 'module.' if present)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    print("Model loaded.")

    # 3. Find Targets in Dataset
    # We search in test, val, then train
    files = ["test_data.pth", "val_data.pth", "train_data.pth"]
    found_data = {}

    for fname in files:
        fpath = os.path.join(DATASET_DIR, fname)
        if not os.path.exists(fpath):
            continue
            
        print(f"Scanning {fname}...")
        try:
            # map_location='cpu' to avoid cuda errors if saved on gpu
            dataset = torch.load(fpath, map_location='cpu')
            for data in dataset:
                if hasattr(data, 'mp_id') and data.mp_id in TARGET_IDS:
                    if data.mp_id not in found_data:
                        found_data[data.mp_id] = data
                        print(f"Found {TARGET_IDS[data.mp_id]} ({data.mp_id}) in {fname}")
        except Exception as e:
            print(f"Error loading {fname}: {e}")

    print(f"Found {len(found_data)}/{len(TARGET_IDS)} targets.")

    # 4. Predict
    results = {}
    with torch.no_grad():
        for mp_id, data in found_data.items():
            formula = TARGET_IDS[mp_id]
            
            # Prepare batch
            data = data.to(device)
            # Add batch index if missing (for single graph batch is zeros)
            if not hasattr(data, 'batch') or data.batch is None:
                data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
            
            try:
                # Forward pass
                # CGT_phys forward expects: x, edge_index, edge_attr, energies, elec_conf, orbital_counts
                # The data object from .pth should have these.
                out = model(data) # [1, 4, 201]
                pred_pdos = out.cpu().numpy()[0]
                
                # Smooth prediction
                pred_pdos_smooth = np.zeros_like(pred_pdos)
                for i in range(4):
                    pred_pdos_smooth[i] = smooth(pred_pdos[i])
                
                results[formula] = {
                    "energies": data.energies.cpu().numpy()[0],
                    "real": data.y.cpu().numpy()[0],
                    "pred": pred_pdos_smooth
                }
            except Exception as e:
                print(f"Prediction failed for {formula}: {e}")

    # 5. Plot (Reproduce Figure 4 style)
    plot_results(results)

def plot_results(results):
    if not results:
        print("No results to plot.")
        return

    n_samples = len(results)
    cols = 3
    rows = (n_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    if n_samples == 1: axes = [axes]
    else: axes = axes.flatten()
    
    orbitals = ['s', 'p', 'd'] 
    # Colors matching the paper style (usually blue/orange/green/red)
    colors = {'s': '#1f77b4', 'p': '#ff7f0e', 'd': '#2ca02c'}
    
    sorted_formulas = sorted(results.keys()) # Or specific order?
    # Use order from TARGET_IDS values if possible for consistency
    ordered_formulas = [v for k,v in TARGET_IDS.items() if v in results]
    
    for i, formula in enumerate(ordered_formulas):
        if i >= len(axes): break
        ax = axes[i]
        res = results[formula]
        
        energies = res["energies"]
        real_dos = res["real"]
        pred_dos = res["pred"]
        
        # Clip negative predictions
        pred_dos = np.maximum(pred_dos, 0)
        
        # Offset for stacking
        offset = 0
        for idx, orb in enumerate(orbitals): # 0=s, 1=p, 2=d
            r_d = real_dos[idx]
            p_d = pred_dos[idx]
            
            # Determine spacing based on max value
            current_max = max(r_d.max(), p_d.max())
            spacing = current_max * 0.5 + 0.5 # Heuristic spacing
            
            # Real: Filled area
            ax.fill_between(energies, r_d + offset, offset, color=colors[orb], alpha=0.3)
            ax.plot(energies, r_d + offset, color=colors[orb], linewidth=1, alpha=0.6)
            
            # Pred: Dashed line
            ax.plot(energies, p_d + offset, color=colors[orb], linestyle='--', linewidth=2)
            
            # Update offset
            offset += spacing * 1.2

        ax.set_title(formula, fontsize=14, fontweight='bold')
        ax.set_xlim(-10, 10)
        ax.set_xlabel("Energy (eV)")
        if i % cols == 0:
            ax.set_ylabel("PDOS (a.u.)")
            
        # Legend only on first plot
        if i == 0:
            from matplotlib.lines import Line2D
            custom_lines = [
                Line2D([0], [0], color='gray', alpha=0.3, lw=4, label='Real (DFT)'),
                Line2D([0], [0], color='gray', linestyle='--', lw=2, label='Predicted (CGT)')
            ]
            ax.legend(handles=custom_lines, loc='upper right')

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Plot saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()
