import torch
import os
import sys
import argparse
from torch.utils.data import DataLoader, Subset

# 确保能找到你的模型定义
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from WLY.model import CrystalTransformer
from WLY.data_loader import CrystalGraphDataset, collate_fn

def move_to_device(obj, device, non_blocking=False):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device, non_blocking=non_blocking) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [move_to_device(v, device, non_blocking=non_blocking) for v in obj]
        return type(obj)(converted)
    return obj

def predict_single():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=os.path.join(base_dir, "checkpoints", "latest_model.pth"))
    parser.add_argument("--data", default=os.path.join(base_dir, "final_dataset.pkl"))
    parser.add_argument("--features", default=os.path.join(base_dir, "atom_features.pth"))
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 1. 加载 Checkpoint
    checkpoint = torch.load(args.ckpt, map_location=device)
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
    dataset = CrystalGraphDataset(args.data, args.features, device="cpu")

    num = min(args.num, len(dataset))
    subset = Subset(dataset, range(num))
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.pin_memory or device.type == "cuda"),
        collate_fn=collate_fn,
    )

    print(f"\n--- Testing on first {num} samples ---")
    sample_idx = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device, non_blocking=(device.type == "cuda"))
            pred_norm = model(batch)

            preds = pred_norm.detach()
            preds_denorm = preds * norm_stats["std"] + norm_stats["mean"]
            targets = batch["target"]

            for j in range(preds_denorm.shape[0]):
                prediction = float(preds_denorm[j].item())
                target = float(targets[j].item())
                error = abs(prediction - target)
                print(f"Sample {sample_idx+1}:")
                print(f"  > Real Energy: {target:.4f} eV")
                print(f"  > Pred Energy: {prediction:.4f} eV")
                print(f"  > Error:       {error:.4f} eV")
                print("-" * 30)
                sample_idx += 1

if __name__ == "__main__":
    predict_single()
