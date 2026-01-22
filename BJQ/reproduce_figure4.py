import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# --- Configuration ---
RESULTS_CSV = r'd:\Github hanjia\whuphy-attention\BJQ\prediction_results.csv'
PDOS_JSON = r'd:\Github hanjia\whuphy-attention\BJQ\prediction_pdos_sample.json'
OUTPUT_DIR = r'd:\Github hanjia\whuphy-attention\BJQ'
# Energy range for PDOS (assumed from -10 to 10 eV, 201 points)
ENERGIES = np.linspace(-10, 10, 201)

def plot_figure4_reproduction(pdos_data):
    """
    Reproduces Figure 4 style: 6 sample PDOS plots (Predicted vs. Truth)
    Since we only have PREDICTIONS (no ground truth in DB for this task),
    we will plot only the PREDICTED PDOS lines.
    
    Structure: 2 rows x 3 columns
    """
    if not pdos_data:
        print("No PDOS data found for plotting.")
        return

    mp_ids = list(pdos_data.keys())[:6] # Take up to 6
    num_plots = len(mp_ids)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    orbitals = ['s', 'p', 'd', 'f']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Standard Tableau colors
    
    for i, mp_id in enumerate(mp_ids):
        if i >= 6: break
        
        ax = axes[i]
        pdos = np.array(pdos_data[mp_id]) # [4, 201]
        
        # Plot each orbital
        for orb_idx in range(3): # Plot s, p, d (skip f to avoid clutter, or plot all)
            ax.plot(ENERGIES, pdos[orb_idx], label=f'{orbitals[orb_idx]}-pred', 
                    color=colors[orb_idx], linewidth=1.5)
            
            # If we had Ground Truth, we would fill_between:
            # ax.fill_between(ENERGIES, pdos_gt[orb_idx], alpha=0.3, color=colors[orb_idx])
            
        ax.set_title(f'Sample: {mp_id}', fontsize=10)
        ax.set_xlim(-10, 10)
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('PDOS (states/eV)')
        
        # Only add legend to the first plot
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
            
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'reproduced_figure4.png'), dpi=300)
    print(f"Figure 4 saved to {os.path.join(OUTPUT_DIR, 'reproduced_figure4.png')}")
    plt.close()

def plot_figure5_reproduction(df):
    """
    Reproduces Figure 5 style: Scatter plots for scalar properties.
    Now we try to plot "Predicted vs Theoretical" (if Ground Truth is non-zero).
    
    We filter out samples where Ground Truth is 0.0 (likely missing data).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Check if ground truth columns exist
    if 'true_formation_energy' not in df.columns:
        print("Warning: 'true_formation_energy' not found in results. Falling back to histogram.")
        axes[0].hist(df['pred_formation_energy'], bins=50, color='skyblue', edgecolor='black')
        axes[0].set_title('Predicted Formation Energy Distribution')
        axes[0].set_xlabel('Formation Energy (eV/atom)')
    else:
        # --- 1. Formation Energy ---
        # Filter valid ground truth
        df_fe = df[df['true_formation_energy'] != 0.0]
        
        if len(df_fe) > 10:
            # Scatter Plot
            x = df_fe['true_formation_energy']
            y = df_fe['pred_formation_energy']
            axes[0].scatter(x, y, alpha=0.5, color='skyblue', edgecolor='k')
            
            # Reference line
            min_val = min(x.min(), y.min())
            max_val = max(x.max(), y.max())
            axes[0].plot([min_val, max_val], [min_val, max_val], 'r--')
            
            # Metrics
            rmse = np.sqrt(((x - y) ** 2).mean())
            mae = np.abs(x - y).mean()
            axes[0].text(0.05, 0.9, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}', transform=axes[0].transAxes)
            
            axes[0].set_xlabel('DFT Formation Energy (eV/atom)')
            axes[0].set_ylabel('Predicted Formation Energy (eV/atom)')
            axes[0].set_title('Formation Energy: Pred vs DFT')
        else:
            # Fallback to Histogram
            axes[0].hist(df['pred_formation_energy'], bins=50, color='skyblue', edgecolor='black')
            axes[0].set_title('Predicted Formation Energy Distribution (No GT)')
            axes[0].set_xlabel('Formation Energy (eV/atom)')

    if 'true_band_gap' not in df.columns:
        print("Warning: 'true_band_gap' not found in results. Falling back to histogram.")
        axes[1].hist(df['pred_band_gap'], bins=50, color='salmon', edgecolor='black')
        axes[1].set_title('Predicted Band Gap Distribution')
        axes[1].set_xlabel('Band Gap (eV)')
    else:
        # --- 2. Band Gap ---
        df_bg = df[df['true_band_gap'] != 0.0]
        
        if len(df_bg) > 10:
            x = df_bg['true_band_gap']
            y = df_bg['pred_band_gap']
            axes[1].scatter(x, y, alpha=0.5, color='salmon', edgecolor='k')
            
            min_val = min(x.min(), y.min())
            max_val = max(x.max(), y.max())
            axes[1].plot([min_val, max_val], [min_val, max_val], 'r--')
            
            rmse = np.sqrt(((x - y) ** 2).mean())
            mae = np.abs(x - y).mean()
            axes[1].text(0.05, 0.9, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}', transform=axes[1].transAxes)
            
            axes[1].set_xlabel('DFT Band Gap (eV)')
            axes[1].set_ylabel('Predicted Band Gap (eV)')
            axes[1].set_title('Band Gap: Pred vs DFT')
        else:
            axes[1].hist(df['pred_band_gap'], bins=50, color='salmon', edgecolor='black')
            axes[1].set_title('Predicted Band Gap Distribution (No GT)')
            axes[1].set_xlabel('Band Gap (eV)')
    
    # --- 3. p-band Center ---
    # We don't have p-band center GT in DB, so always Histogram
    axes[2].hist(df['pred_p_band_center'], bins=50, color='lightgreen', edgecolor='black')
    axes[2].set_title('Predicted p-band Center Distribution')
    axes[2].set_xlabel('p-band Center (eV)')
    axes[2].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'reproduced_figure5.png'), dpi=300)
    print(f"Figure 5 saved to {os.path.join(OUTPUT_DIR, 'reproduced_figure5.png')}")
    plt.close()

def run_plotting():
    # Load Data
    if os.path.exists(RESULTS_CSV):
        df = pd.read_csv(RESULTS_CSV)
        plot_figure5_reproduction(df)
    else:
        print("Results CSV not found. Run prediction first.")
        
    if os.path.exists(PDOS_JSON):
        with open(PDOS_JSON, 'r') as f:
            pdos_data = json.load(f)
        plot_figure4_reproduction(pdos_data)
    else:
        print("PDOS JSON not found. Run prediction first.")

if __name__ == '__main__':
    run_plotting()
