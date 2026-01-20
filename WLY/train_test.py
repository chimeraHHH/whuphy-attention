import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from WLY.data_loader import CrystalGraphDataset, collate_fn
from WLY.model import CrystalTransformer

class Normalizer:
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

def train_one_epoch():
    # --- Config ---
    DATA_PATH = '/Users/wuleyan/Desktop/dachuang/whuphy-attention/WLY/final_dataset.pkl'
    FEATURE_PATH = '/Users/wuleyan/Desktop/dachuang/whuphy-attention/WLY/atom_features.pth'
    
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # --- Data Preparation ---
    print("Initializing dataset...")
    full_dataset = CrystalGraphDataset(DATA_PATH, FEATURE_PATH, device=DEVICE)
    
    # Calculate Normalization Stats on FULL dataset (target)
    # We collect all targets first
    all_targets = [sample['target'] for sample in full_dataset.data]
    target_tensor = torch.tensor(all_targets, dtype=torch.float32, device=DEVICE)
    normalizer = Normalizer(target_tensor)
    print(f"Target Normalization: Mean={normalizer.mean:.4f}, Std={normalizer.std:.4f}")
    
    # For this quick test, we only use a small subset (enough for 2 batches)
    # 2 batches * 32 = 64 samples
    subset_indices = range(64)
    small_dataset = Subset(full_dataset, subset_indices)
    
    loader = DataLoader(
        small_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # --- Model Setup ---
    print("Initializing model...")
    model = CrystalTransformer(
        atom_fea_len=9, 
        hidden_dim=128, 
        n_local_layers=3, 
        n_global_layers=2
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() # Regression task (eform)
    
    # --- Training Loop ---
    print("\nStarting training (1 epoch, 2 batches)...")
    model.train()
    
    start_time = time.time()
    total_loss = 0
    
    for i, batch in enumerate(loader):
        # Forward
        preds = model(batch)
        targets = batch['target']
        
        # Normalize targets
        targets_norm = normalizer.norm(targets)
        
        loss = criterion(preds, targets_norm)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        current_loss = loss.item()
        total_loss += current_loss
        
        # Calculate MAE in original scale
        with torch.no_grad():
            preds_denorm = normalizer.denorm(preds)
            mae = torch.abs(preds_denorm - targets).mean().item()
            
        print(f"Batch {i+1}: Loss (Norm) = {current_loss:.4f} | MAE (Orig) = {mae:.4f}")
        
        # Stop after 2 batches as requested
        if i + 1 >= 2:
            break
            
    avg_loss = total_loss / 2
    print(f"\nTraining finished in {time.time() - start_time:.2f}s")
    print(f"Average Loss (MSE): {avg_loss:.4f}")

if __name__ == "__main__":
    train_one_epoch()
