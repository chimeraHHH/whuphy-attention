import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

# Paths
work_dir = r"D:\Github hanjia\whuphy-attention\BJQ"
pred_path = os.path.join(work_dir, "predictions_full.csv")
history_path = os.path.join(work_dir, "training_history.csv")
output_path_pdf = os.path.join(work_dir, "reproduced_figure4_results.pdf")
output_path_png = os.path.join(work_dir, "final_result_plot.png")

print(f"Reading data from {pred_path} and {history_path}...")

# Load data
try:
    pred_df = pd.read_csv(pred_path)
    history_df = pd.read_csv(history_path)
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Calculate metrics
# Remove NaN if any
pred_df = pred_df.dropna()
actual = pred_df['Actual']
predicted = pred_df['Predicted']

r2 = r2_score(actual, predicted)
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))

print(f"Metrics: R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

# Create figure
fig = plt.figure(figsize=(14, 6))

# --- Subplot 1: Parity Plot (Predicted vs Actual) ---
ax1 = fig.add_subplot(1, 2, 1)

# Scatter plot with color mapping based on density or error could be nice, but simple scatter is fine
# Using absolute error for color
error = np.abs(actual - predicted)
sc = ax1.scatter(actual, predicted, c=error, cmap='viridis', alpha=0.7, s=50, edgecolors='w', linewidth=0.5)
plt.colorbar(sc, ax=ax1, label='Absolute Error (eV)')

# Diagonal line
min_val = min(actual.min(), predicted.min())
max_val = max(actual.max(), predicted.max())
# Add some padding
range_val = max_val - min_val
plot_min = min_val - range_val * 0.05
plot_max = max_val + range_val * 0.05

ax1.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', lw=2, label='Ideal Parity')

# Statistics text
stats_text = f"$R^2 = {r2:.3f}$\nMAE = {mae:.3f} eV\nRMSE = {rmse:.3f} eV"
ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax1.set_xlabel('DFT Calculated Formation Energy (eV)', fontweight='bold')
ax1.set_ylabel('ML Predicted Formation Energy (eV)', fontweight='bold')
ax1.set_title('Defect Formation Energy Prediction', fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_aspect('equal', adjustable='datalim')

# --- Subplot 2: Training History ---
ax2 = fig.add_subplot(1, 2, 2)

if not history_df.empty:
    epochs = history_df['epoch']
    train_loss = history_df['train_loss_mse']
    val_mae = history_df['val_mae']
    
    # Primary axis: Train Loss
    line1 = ax2.plot(epochs, train_loss, 'b-', lw=2, label='Train Loss (MSE)')
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Training Loss (MSE)', color='b', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='b')
    
    # Secondary axis: Val MAE
    ax2_twin = ax2.twinx()
    line2 = ax2_twin.plot(epochs, val_mae, 'r-', lw=2, label='Validation MAE')
    ax2_twin.set_ylabel('Validation MAE (eV)', color='r', fontweight='bold')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right')
    
    ax2.set_title('Training Dynamics', fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
else:
    ax2.text(0.5, 0.5, 'No training history available', ha='center', va='center')

plt.tight_layout()

# Save
print(f"Saving plot to {output_path_pdf} and {output_path_png}...")
plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
print("Done.")
