
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error

# Import model and data utils
from models.Final_Project_Model import HierarchicalHybridMatformer
from data import CIFData # Needed for unpickling if class is used

# Set plotting style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def load_best_model(model_path, device):
    model = HierarchicalHybridMatformer(
        node_input_dim=118,
        edge_dim=128,
        hidden_dim=256,
        out_dim=201*4,
        num_layers=3,
        heads=4,
        dropout=0.1
    )
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model file {model_path} not found!")
    
    model.to(device)
    model.eval()
    return model

def plot_pdos_comparison(y_true, y_pred, energies, save_path, sample_indices=None):
    """
    Plot PDOS comparison for selected samples (Figure 4 style).
    y_true/y_pred shape: [N, 4, 201] (s, p, d, f)
    """
    if sample_indices is None:
        # Select 3 random samples
        sample_indices = np.random.choice(len(y_true), 3, replace=False)
    
    orbitals = ['s', 'p', 'd', 'f']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    fig, axes = plt.subplots(len(sample_indices), 4, figsize=(16, 3*len(sample_indices)), sharex=True, sharey='row')
    
    for row_idx, sample_idx in enumerate(sample_indices):
        for col_idx, orbital in enumerate(orbitals):
            ax = axes[row_idx, col_idx] if len(sample_indices) > 1 else axes[col_idx]
            
            # Ground Truth
            ax.plot(energies, y_true[sample_idx, col_idx], 
                   color='black', linestyle='--', alpha=0.6, linewidth=1.5, label='DFT (True)')
            
            # Prediction
            ax.plot(energies, y_pred[sample_idx, col_idx], 
                   color=colors[col_idx], alpha=0.9, linewidth=1.5, label=f'Pred ({orbital})')
            
            # Fill between
            ax.fill_between(energies, y_pred[sample_idx, col_idx], 0, 
                           color=colors[col_idx], alpha=0.2)
            
            if row_idx == 0:
                ax.set_title(f'{orbital}-orbital PDOS', fontweight='bold')
            if row_idx == len(sample_indices) - 1:
                ax.set_xlabel('Energy (eV)')
            if col_idx == 0:
                ax.set_ylabel(f'Sample {sample_idx}\nDensity of States')
                
            ax.legend(loc='upper right', fontsize='small')
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved PDOS plot to {save_path}")
    plt.close()

def plot_scalar_parity(y_true, y_pred, property_name, unit, save_path):
    """
    Plot parity plot for scalar properties (Figure 5 style).
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    plt.figure(figsize=(6, 6))
    
    # Scatter points
    plt.scatter(y_true, y_pred, alpha=0.5, s=15, c='#2b83ba', edgecolors='none')
    
    # Reference line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.05
    
    plt.plot([min_val-margin, max_val+margin], [min_val-margin, max_val+margin], 
            'k--', alpha=0.5, label='Ideal')
    
    plt.xlabel(f'DFT {property_name} ({unit})')
    plt.ylabel(f'Predicted {property_name} ({unit})')
    plt.title(f'{property_name} Prediction')
    
    # Statistics box
    stats = (f'$R^2 = {r2:.3f}$\n'
             f'MAE = {mae:.3f} {unit}')
    plt.text(0.05, 0.95, stats, transform=plt.gca().transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlim(min_val-margin, max_val+margin)
    plt.ylim(min_val-margin, max_val+margin)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved parity plot to {save_path}")
    plt.close()

def main():
    root_dir = r"d:\Github hanjia\whuphy-attention\BJQ\GNNs-PDOS"
    data_path = os.path.join(root_dir, "shujujione", "JPCA2025", "dataset", "test_data.pth")
    model_path = os.path.join(root_dir, "best_multitask_model.pth")
    # Also check for final model if best doesn't exist yet
    if not os.path.exists(model_path):
        model_path = os.path.join(root_dir, "final_multitask_model.pth")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    print("Loading test data...")
    if not os.path.exists(data_path):
        print(f"Test data not found at {data_path}")
        return
        
    dataset = torch.load(data_path)
    # We assume dataset is already enriched or we just plot what we have.
    # For scalars, if they are 0.0, the plot will show that.
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 2. Load Model
    model = load_best_model(model_path, device)
    
    # 3. Inference
    print("Running inference...")
    all_pred_pdos = []
    all_true_pdos = []
    all_pred_scalar = []
    all_true_scalar = [] # [Formation Energy, Band Gap, p-band Center]
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred_pdos, pred_scalar = model(batch)
            
            all_pred_pdos.append(pred_pdos.cpu().numpy())
            all_true_pdos.append(batch.y.cpu().numpy())
            
            # Check if scalar properties exist in batch, otherwise use zeros
            if hasattr(batch, 'formation_energy') and hasattr(batch, 'band_gap'):
                true_scalar = torch.stack([
                    batch.formation_energy.squeeze(), 
                    batch.band_gap.squeeze(), 
                    batch.p_band_center.squeeze()
                ], dim=1)
            else:
                # Fallback if enrichment didn't happen
                true_scalar = torch.stack([
                    torch.zeros_like(batch.p_band_center.squeeze()),
                    torch.zeros_like(batch.p_band_center.squeeze()),
                    batch.p_band_center.squeeze()
                ], dim=1)
                
            all_true_scalar.append(true_scalar.cpu().numpy())
            all_pred_scalar.append(pred_scalar.cpu().numpy())
            
    # Concatenate
    y_pred_pdos = np.concatenate(all_pred_pdos, axis=0)
    y_true_pdos = np.concatenate(all_true_pdos, axis=0)
    y_pred_scalar = np.concatenate(all_pred_scalar, axis=0)
    y_true_scalar = np.concatenate(all_true_scalar, axis=0)
    
    # 4. Plotting
    print("Generating plots...")
    
    # Energy grid (from -6 to 6 eV usually, or extract from data)
    # Assuming standard grid from previous files
    energies = np.linspace(-6, 6, 201) 
    
    # Figure 4: PDOS
    plot_pdos_comparison(y_true_pdos, y_pred_pdos, energies, 
                        os.path.join(output_dir, "reproduced_figure4_multitask.png"))
    
    # Figure 5: Scalar Parity Plots
    # Formation Energy
    if np.any(y_true_scalar[:, 0] != 0):
        plot_scalar_parity(y_true_scalar[:, 0], y_pred_scalar[:, 0], 
                          "Formation Energy", "eV", 
                          os.path.join(output_dir, "parity_formation_energy.png"))
    
    # Band Gap
    if np.any(y_true_scalar[:, 1] != 0):
        plot_scalar_parity(y_true_scalar[:, 1], y_pred_scalar[:, 1], 
                          "Band Gap", "eV", 
                          os.path.join(output_dir, "parity_band_gap.png"))
                          
    # p-band Center
    plot_scalar_parity(y_true_scalar[:, 2], y_pred_scalar[:, 2], 
                      "p-band Center", "eV", 
                      os.path.join(output_dir, "parity_p_band_center.png"))

    # 5. Save Predictions to CSV
    print("Saving predictions to CSV...")
    df = pd.DataFrame({
        'True_E_form': y_true_scalar[:, 0],
        'Pred_E_form': y_pred_scalar[:, 0],
        'True_Gap': y_true_scalar[:, 1],
        'Pred_Gap': y_pred_scalar[:, 1],
        'True_p_Center': y_true_scalar[:, 2],
        'Pred_p_Center': y_pred_scalar[:, 2]
    })
    df.to_csv(os.path.join(output_dir, "predictions_multitask.csv"), index=False)
    print("Done.")

if __name__ == "__main__":
    main()
