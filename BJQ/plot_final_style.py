import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
CSV_PATH = r'd:\Github hanjia\whuphy-attention\BJQ\test_set_predictions.csv'
OUTPUT_DIR = r'd:\Github hanjia\whuphy-attention\BJQ'
OUTPUT_FILE = 'final_figure5_600test_cleaned.png'

def plot_final_style():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    df = pd.read_csv(CSV_PATH)
    
    # Filter outliers to focus on the main distribution (similar to the reference image range -10 to 20)
    # The user specifically complained about outliers in the 600-sample plot (-150, 200).
    # We will strictly limit to [-20, 30] to show the dense cluster.
    
    mask = (df['true_formation_energy'] > -20) & (df['true_formation_energy'] < 30) & \
           (df['pred_formation_energy'] > -20) & (df['pred_formation_energy'] < 30)
    
    df_filtered = df[mask]
    print(f"Original samples: {len(df)}, Filtered samples ([-20, 30]): {len(df_filtered)}")
    
    x = df_filtered['true_formation_energy']
    y = df_filtered['pred_formation_energy']
    
    # Metrics
    mae = np.abs(x - y).mean()
    
    plt.figure(figsize=(8, 8))
    
    # Scatter plot with reference style
    # Blue dots (#6da9cf is similar to the light blue in image), white edge
    plt.scatter(x, y, alpha=0.8, c='#6da9cf', edgecolors='white', linewidth=0.8, s=60, label='Model Predictions')
    
    # Ideal line
    # Set fixed limits for better visualization consistency
    plot_min = -15
    plot_max = 25
    
    plt.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', linewidth=2.5, label='Ideal (Real=Pred)')
    
    plt.xlim(plot_min, plot_max)
    plt.ylim(plot_min, plot_max)
    
    # Labels and Title
    plt.xlabel('Ground Truth Energy (eV)', fontsize=14)
    plt.ylabel('Predicted Energy (eV)', fontsize=14)
    plt.title(f'Crystal Energy Prediction (MAE: {mae:.4f} eV)', fontsize=16)
    
    # Grid and Legend
    plt.grid(True, linestyle='-', alpha=0.15, color='gray')
    plt.legend(loc='upper left', fontsize=12, frameon=True, edgecolor='gray')
    
    # Tick params
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    plot_final_style()
