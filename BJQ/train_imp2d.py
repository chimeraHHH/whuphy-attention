import os
print("Script started...")
import pandas as pd
import numpy as np
import sys
print("Standard libs imported...")
import torch
import torch.nn as nn
print("Torch imported...")
import torch.optim as optim
from torch_geometric.loader import DataLoader
print("Geometric imported...")
# from tqdm import tqdm
# import matplotlib.pyplot as plt
print("All imports done.")

# Add path to import model
sys.path.append(r'd:\Github hanjia\whuphy-attention\BJQ\GNNs-PDOS\models')
from Final_Project_Model import HierarchicalHybridMatformer
from data_inference import Imp2DDataset

# --- Configuration ---
DB_PATH = r'd:\Github hanjia\whuphy-attention\BJQ\imp2d.db'
ORB_PATH = r'd:\Github hanjia\whuphy-attention\BJQ\orbital_electrons.json'
# We start from the pre-trained weights to leverage the learned structural features
PRETRAINED_WEIGHTS = r'd:\Github hanjia\whuphy-attention\BJQ\GNNs-PDOS\shujujione\JPCA2025\CGT_phys_weights.pth'
OUTPUT_DIR = r'd:\Github hanjia\whuphy-attention\BJQ'
BATCH_SIZE = 32
EPOCHS = 50 # Set to 50 Epochs
LR = 1e-4
DEVICE = torch.device('cpu') # Force CPU to avoid crashes

def train_and_evaluate():
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data
    print("Loading dataset from DB (filtering for valid formation energy)...")
    dataset_handler = Imp2DDataset(DB_PATH, ORB_PATH)
    # Load ALL data first, then sample. Or load with limit if load_all supports random sampling.
    # Current load_all takes 'limit' which is just first N. 
    # To be random, we better load more then sample, or modify load_all.
    # Given DB size is ~17k, loading all meta-data is fast, but graph construction takes time.
    # Let's modify strategy: Load a larger chunk (e.g. 2000) then randomly select 800 valid ones.
    # Or just load all (it took ~30s before?) and then sample.
    # If graph construction is the bottleneck, we should sample indices first.
    # But Imp2DDataset structure couples row processing with loading.
    # Let's just sample from the loaded list for now, assuming loading 17k rows isn't the main bottleneck,
    # but training 17k * 50 epochs is.
    
    # Actually, constructing 17k graphs might take 10-20 mins.
    # Optimization: Only load a random subset from DB directly if possible, or use a limit.
    # We want to maximize the number of points in the plot.
    # Load ALL data from DB.
    all_data = dataset_handler.load_all(limit=None) 
    
    # Filter out samples with no valid ground truth
    # (Already filtered in dataset_handler.process_row for converged & eform)
    
    print(f"Total valid samples loaded (pool): {len(all_data)}")
    
    # --- Sampling Strategy ---
    # Strategy: Load a large pool for VISUALIZATION DENSITY, but use a small subset for TRAINING SPEED.
    # We load 8000 samples total.
    # Train set: ~1500 (Fast epochs)
    # Test set: ~6500 (Dense plot)
    
    # We want to maximize the number of points in the plot.
    # Load ALL data from DB.
    all_data = dataset_handler.load_all(limit=8000) 
    
    # Filter out samples with no valid ground truth
    # (Already filtered in dataset_handler.process_row for converged & eform)
    
    # --- Data Cleaning ---
    # Filter out physical outliers to improve training stability
    # Keep only samples within [-20, 20] eV
    cleaned_data = []
    for data in all_data:
        fe = data.formation_energy.item()
        if -20.0 <= fe <= 20.0:
            cleaned_data.append(data)
    
    print(f"Total valid samples loaded (pool): {len(all_data)}")
    print(f"Samples after outlier cleaning ([-20, 20] eV): {len(cleaned_data)}")
    all_data = cleaned_data
    
    # Check if we have enough data
    if len(all_data) < 50:
        print("Warning: Too few samples for meaningful training/plotting.")
    
    # Split Data
    # Strategy: 70% Train (to learn well), 30% Test (for dense plot)
    # Total valid pool is ~8000. 
    # Train ~5600 (Good size for convergence)
    # Test ~2400 (Very dense plot)
    
    import random
    random.seed(42)
    random.shuffle(all_data)
    
    n_total = len(all_data)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.05) # Small validation set is enough
    n_test = n_total - n_train - n_val
    
    train_data = all_data[:n_train]
    val_data = all_data[n_train:n_train+n_val]
    test_data = all_data[n_train+n_val:]
    
    print(f"Split Strategy: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # --- Target Normalization ---
    # Compute Mean and Std from TRAIN set only
    train_fe_values = [d.formation_energy.item() for d in train_data]
    fe_mean = np.mean(train_fe_values)
    fe_std = np.std(train_fe_values)
    print(f"Target Normalization Stats: Mean={fe_mean:.4f}, Std={fe_std:.4f}")
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Initialize Model
    model = HierarchicalHybridMatformer(
        node_input_dim=118,
        edge_dim=128,
        hidden_dim=256,
        out_dim=201*4,
        num_layers=3,
        heads=4,
        dropout=0.1,
        use_vacancy_feature=True
    ).to(DEVICE)
    
    # Load Pre-trained Weights (Partial)
    try:
        state_dict = torch.load(PRETRAINED_WEIGHTS, map_location=DEVICE)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
        print("Pre-trained weights loaded (partial).")
    except Exception as e:
        print(f"Warning: Could not load pre-trained weights: {e}")
        print("Starting from scratch.")

    # 3. Training Setup
    # Optimization Strategy for Lower MAE:
    # 1. Loss: Switch to L1Loss (Mean Absolute Error) to directly optimize the target metric.
    #    MSE focuses on outliers, L1 focuses on the median and reduces MAE effectively.
    # 2. Scheduler: Reduce LR when validation metric plateaus.
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.L1Loss() # Changed from MSELoss to L1Loss
    
    # Scheduler: Reduce LR by factor of 0.5 if Val MAE doesn't improve for 5 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 4. Training Loop
    best_val_mae = float('inf')
    history = {'train_loss': [], 'val_mae': []}
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0
        
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            
            # Forward
            _, out_scalar = model(batch)
            
            # Prediction: index 0 is Formation Energy
            pred_fe = out_scalar[:, 0]
            true_fe = batch.formation_energy.squeeze()
            
            # Normalize Target
            true_fe_norm = (true_fe - fe_mean) / fe_std
            
            loss = criterion(pred_fe, true_fe_norm)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item() * batch.num_graphs
            
        avg_train_loss = train_loss_sum / len(train_data)
        
        # Validation
        model.eval()
        val_mae_sum = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                _, out_scalar = model(batch)
                pred_fe_norm = out_scalar[:, 0]
                true_fe = batch.formation_energy.squeeze()
                
                # Denormalize Prediction for MAE calculation
                pred_fe = pred_fe_norm * fe_std + fe_mean
                
                val_mae_sum += torch.abs(pred_fe - true_fe).sum().item()
        
        avg_val_mae = val_mae_sum / len(val_data)
        
        # Step the scheduler
        scheduler.step(avg_val_mae)
        
        history['train_loss'].append(avg_train_loss)
        history['val_mae'].append(avg_val_mae)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss (L1): {avg_train_loss:.4f} | Val MAE: {avg_val_mae:.4f}")
        
        # Save Best Model
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_finetuned_model.pth'))
    
    print("Training complete.")
    
    # 5. Final Inference on Test Set
    print("Running inference on Test Set with Best Model...")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_finetuned_model.pth')))
    model.eval()
    
    results = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            _, out_scalar = model(batch)
            
            for i in range(batch.num_graphs):
                mp_id = batch.mp_id[i]
                pred_fe_norm = out_scalar[i, 0].item()
                true_fe = batch.formation_energy[i].item()
                
                # Denormalize Prediction
                pred_fe = pred_fe_norm * fe_std + fe_mean
                
                results.append({
                    'mp_id': mp_id,
                    'pred_formation_energy': pred_fe,
                    'true_formation_energy': true_fe
                })
                
    # Save Results
    df = pd.DataFrame(results)
    output_csv = os.path.join(OUTPUT_DIR, 'test_set_predictions.csv')
    df.to_csv(output_csv, index=False)
    print(f"Test results saved to {output_csv}")
    
    # 6. Plotting (Directly here to ensure consistency)
    # plot_scatter(df)

# def plot_scatter(df):
#     plt.figure(figsize=(8, 8)) # Square figure often looks better for Real vs Pred
#     x = df['true_formation_energy']
#     y = df['pred_formation_energy']
#     
#     # Metrics
#     rmse = np.sqrt(((x - y) ** 2).mean())
#     mae = np.abs(x - y).mean()
#     r2 = np.corrcoef(x, y)[0, 1]**2
#     
#     # Style imitation: Blue dots with white edge, slightly transparent
#     plt.scatter(x, y, alpha=0.7, c='#6da9cf', edgecolors='white', s=50, label='Model Predictions')
#     
#     # Reference line
#     min_val = min(x.min(), y.min())
#     max_val = max(x.max(), y.max())
#     # Extend range slightly for better view
#     range_span = max_val - min_val
#     plot_min = min_val - range_span * 0.05
#     plot_max = max_val + range_span * 0.05
#     
#     plt.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', linewidth=2, label='Ideal (Real=Pred)')
#     
#     plt.xlim(plot_min, plot_max)
#     plt.ylim(plot_min, plot_max)
#     
#     plt.xlabel('Ground Truth Energy (eV)', fontsize=14)
#     plt.ylabel('Predicted Energy (eV)', fontsize=14)
#     plt.title(f'Crystal Energy Prediction (MAE: {mae:.4f} eV)', fontsize=16)
#     
#     plt.legend(loc='upper left', fontsize=12)
#     plt.grid(True, linestyle='-', alpha=0.2) # Faint grid
#     
#     output_png = os.path.join(OUTPUT_DIR, 'final_figure5_optimized_split.png')
#     plt.savefig(output_png, dpi=300)
#     print(f"Plot saved to {output_png}")

if __name__ == '__main__':
    train_and_evaluate()
