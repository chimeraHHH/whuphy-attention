import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import sys
import os
import time
import numpy as np
from tqdm import tqdm
import gc

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

def main():
    # --- Hyperparameters ---
    CONFIG = {
        'data_path': '/Users/wuleyan/Desktop/dachuang/whuphy-attention/WLY/final_dataset.pkl',
        'feature_path': '/Users/wuleyan/Desktop/dachuang/whuphy-attention/WLY/atom_features.pth',
        'output_dir': '/Users/wuleyan/Desktop/dachuang/whuphy-attention/WLY/checkpoints',
        'batch_size': 16,      
        'epochs': 25,
        'lr': 1e-4,            
        'hidden_dim': 64,      
        'n_local': 2,
        'n_global': 1,
        'seed': 42,
        'resume_path': '/Users/wuleyan/Desktop/dachuang/whuphy-attention/WLY/checkpoints/latest_model.pth' # 如果想从头跑，设为 None
    }
    
    if not os.path.exists(CONFIG['output_dir']):
        os.makedirs(CONFIG['output_dir'])
        
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    # 强制清理起始显存
    if device.type == 'mps':
        torch.mps.empty_cache()

    # --- 1. Load Data ---
    print("Loading dataset...")
    full_dataset = CrystalGraphDataset(CONFIG['data_path'], CONFIG['feature_path'], device=device)
    
    all_targets = [sample['target'] for sample in full_dataset.data]
    target_tensor = torch.tensor(all_targets, dtype=torch.float32, device=device)
    normalizer = Normalizer(target_tensor)
    print(f"Target Norm Stats: Mean={normalizer.mean:.4f}, Std={normalizer.std:.4f}")
    
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    # --- 2. Initialize Model & Optimizer ---
    model = CrystalTransformer(
        atom_fea_len=9,
        hidden_dim=CONFIG['hidden_dim'],
        n_local_layers=CONFIG['n_local'],
        n_global_layers=CONFIG['n_global']
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()
    
    start_epoch = 0
    best_val_mae = float('inf')

    # --- 3. Resume Training (断点续传逻辑) ---
    if CONFIG['resume_path'] and os.path.exists(CONFIG['resume_path']):
        print(f"Resuming from checkpoint: {CONFIG['resume_path']}")
        checkpoint = torch.load(CONFIG['resume_path'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_mae = checkpoint['val_mae']
        print(f"Restarting from Epoch {start_epoch+1}, Best Val MAE was: {best_val_mae:.4f}")

    print("\nStart Training...")
    
    for epoch in range(start_epoch, CONFIG['epochs']):
        start_time = time.time()
        
        # --- Train ---
        model.train()
        train_loss_sum = 0
        train_mae_sum = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", unit="batch")
        
        for batch in train_pbar:
            optimizer.zero_grad()
            preds = model(batch)
            targets = batch['target']
            
            targets_norm = normalizer.norm(targets)
            loss = criterion(preds, targets_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 💡 关键：每个 batch 结束后立即释放显存，防止堆积
            if device.type == 'mps':
                torch.mps.empty_cache()
            
            batch_loss = loss.item()
            train_loss_sum += batch_loss * targets.size(0)
            with torch.no_grad():
                preds_denorm = normalizer.denorm(preds)
                mae = torch.abs(preds_denorm - targets).mean().item()
                train_mae_sum += mae * targets.size(0)
            
            train_pbar.set_postfix({'loss': f'{batch_loss:.4f}', 'mae': f'{mae:.4f}'})
        
        avg_train_loss = train_loss_sum / len(train_set)
        avg_train_mae = train_mae_sum / len(train_set)
        
        # --- Validation ---
        model.eval()
        val_loss_sum = 0
        val_mae_sum = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Val]", unit="batch")
        
        with torch.no_grad():
            for batch in val_pbar:
                preds = model(batch)
                targets = batch['target']
                targets_norm = normalizer.norm(targets)
                loss = criterion(preds, targets_norm)
                val_loss_sum += loss.item() * targets.size(0)
                
                preds_denorm = normalizer.denorm(preds)
                mae = torch.abs(preds_denorm - targets).mean().item()
                val_mae_sum += mae * targets.size(0)
                
                if device.type == 'mps':
                    torch.mps.empty_cache()
                
        avg_val_loss = val_loss_sum / len(val_set)
        avg_val_mae = val_mae_sum / len(val_set)
        
        scheduler.step(avg_val_mae)
        
        # --- 1. 准备存档数据 ---
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'normalizer': {'mean': normalizer.mean, 'std': normalizer.std},
            'config': CONFIG,
            'val_mae': avg_val_mae
        }
        
        # --- 2. 强制保存最新进度（用于接力，不管好坏都要存） ---
        latest_path = os.path.join(CONFIG['output_dir'], 'latest_model.pth')
        torch.save(checkpoint_data, latest_path)

        # --- 3. 只有破纪录时，才保存为最好的模型 ---
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            best_path = os.path.join(CONFIG['output_dir'], 'best_model.pth')
            torch.save(checkpoint_data, best_path)
            saved_msg = "(*)"
        else:
            saved_msg = ""
            
        # 强制 Python 垃圾回收
        gc.collect()
            
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | "
              f"Train Loss: {avg_train_loss:.4f} MAE: {avg_train_mae:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} MAE: {avg_val_mae:.4f} | "
              f"Time: {epoch_time:.1f}s {saved_msg}")

        # 💡 就在这里！加上下面这两行：
        print("🚩 这一轮跑完啦，我要自杀重启来清理内存了...")
        break  # 强制跳出循环，结束 Python 进程

    print("\nTraining Complete.")
    print(f"Best Validation MAE: {best_val_mae:.4f} eV")

if __name__ == "__main__":
    main()