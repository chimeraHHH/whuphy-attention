import torch
import os
import sys

# 确保能找到你的模型定义
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from WLY.model import CrystalTransformer
from WLY.data_loader import CrystalGraphDataset, collate_fn

def predict_single():
    # --- 配置 ---
    CHECKPOINT_PATH = '/Users/wuleyan/Desktop/dachuang/whuphy-attention/WLY/checkpoints/best_model.pth'
    DATA_PATH = '/Users/wuleyan/Desktop/dachuang/whuphy-attention/WLY/final_dataset.pkl'
    FEATURE_PATH = '/Users/wuleyan/Desktop/dachuang/whuphy-attention/WLY/atom_features.pth'
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载 Checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    config = checkpoint['config']
    norm_stats = checkpoint['normalizer']
    
    print(f"Loading model from epoch {checkpoint['epoch']+1} with Val MAE: {checkpoint['val_mae']:.4f}")

    # 2. 初始化模型并加载权重
    model = CrystalTransformer(
        atom_fea_len=9,
        hidden_dim=config['hidden_dim'],
        n_local_layers=config['n_local'],
        n_global_layers=config['n_global']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 3. 加载数据集（用于挑选一个例子做验证）
    dataset = CrystalGraphDataset(DATA_PATH, FEATURE_PATH, device=device)
    
    # 随机选前 5 个来看看
    print("\n--- Testing on first 5 samples ---")
    with torch.no_grad():
        for i in range(5):
            sample = dataset[i]
            # 这里的 batch 需要包装一下，因为模型期待的是 DataLoader 的输出格式
            batch = collate_fn([sample])
            
            # 预测 (得到的是标准化后的值)
            pred_norm = model(batch)
            
            # 反标准化 (Denorm)
            prediction = pred_norm.item() * norm_stats['std'].item() + norm_stats['mean'].item()
            target = sample['target'].item()
            
            error = abs(prediction - target)
            
            print(f"Sample {i+1}:")
            print(f"  > Real Energy: {target:.4f} eV")
            print(f"  > Pred Energy: {prediction:.4f} eV")
            print(f"  > Error:       {error:.4f} eV")
            print("-" * 30)

if __name__ == "__main__":
    predict_single()