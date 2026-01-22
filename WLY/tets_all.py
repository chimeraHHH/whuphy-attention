import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from WLY.data_loader import CrystalGraphDataset, collate_fn
from WLY.model import CrystalTransformer

def move_to_device(obj, device, non_blocking=False):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device, non_blocking=non_blocking) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [move_to_device(v, device, non_blocking=non_blocking) for v in obj]
        return type(obj)(converted)
    return obj

def test_on_full_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=os.path.join(base_dir, "checkpoints", "latest_model.pth"))
    parser.add_argument("--data", default=None)
    parser.add_argument("--features", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--plot_path", default=os.path.join(base_dir, "test_result_plot.png"))
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--top_k", type=int, default=10)
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

    print(f"Loading checkpoint: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location=device)

    config = checkpoint.get("config", {})
    data_path = args.data or config.get("data_path") or os.path.join(base_dir, "final_dataset.pkl")
    feature_path = args.features or config.get("feature_path") or os.path.join(base_dir, "atom_features.pth")
    batch_size = args.batch_size or config.get("batch_size") or 16
    seed = args.seed or config.get("seed") or 42

    print("Loading dataset...")
    full_dataset = CrystalGraphDataset(data_path, feature_path, device="cpu")
    
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    gen = torch.Generator().manual_seed(seed)
    _, _, test_set = random_split(full_dataset, [train_size, val_size, test_size], generator=gen)
    
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.pin_memory or device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=False,
    )

    norm_mean = checkpoint['normalizer']['mean']
    norm_std = checkpoint['normalizer']['std']
    print(f"Loaded Normalizer: Mean={norm_mean:.4f}, Std={norm_std:.4f}")

    hidden_dim = config.get("hidden_dim", 64)
    n_local = config.get("n_local", 2)
    n_global = config.get("n_global", 1)

    model = CrystalTransformer(
        atom_fea_len=9,
        hidden_dim=hidden_dim,
        n_local_layers=n_local,
        n_global_layers=n_global,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_reals = []

    print(f"Starting inference on {len(test_set)} samples...")
    with torch.no_grad():
        for batch in test_loader:
            batch = move_to_device(batch, device, non_blocking=(device.type == "cuda"))
            preds_norm = model(batch)
            targets = batch["target"]
            
            # 反标准化：还原真实物理单位 (eV)
            preds_denorm = preds_norm * norm_std + norm_mean
            
            all_preds.extend(preds_denorm.cpu().numpy().flatten())
            all_reals.extend(targets.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_reals = np.array(all_reals)

    mae = np.mean(np.abs(all_preds - all_reals))
    rmse = np.sqrt(np.mean((all_preds - all_reals)**2))
    
    print("\n" + "="*30)
    print(f"Final Test MAE: {mae:.4f} eV")
    print(f"Final Test RMSE: {rmse:.4f} eV")
    print("="*30)

    plt.figure(figsize=(8, 8))
    plt.scatter(all_reals, all_preds, alpha=0.6, edgecolors='w', label=f'Model Predictions')
    
    # 理想线 y = x
    min_val = min(min(all_reals), min(all_preds))
    max_val = max(max(all_reals), max(all_preds))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal (Real=Pred)')
    
    plt.xlabel('Ground Truth Energy (eV)', fontsize=12)
    plt.ylabel('Predicted Energy (eV)', fontsize=12)
    plt.title(f'Crystal Energy Prediction (MAE: {mae:.4f} eV)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(args.plot_path)
    print(f"\nPlot saved to: {args.plot_path}")
    if args.show:
        plt.show()
    plt.close()

    errors = np.abs(all_preds - all_reals)

    top_k = min(args.top_k, len(errors))
    worst_idx = np.argsort(errors)[-top_k:][::-1]

    print(f"\n======== 误差最大的前 {top_k} 个样本分析 ========")
    print(f"{'排名':<6} | {'索引':<10} | {'真实值(eV)':<12} | {'预测值(eV)':<12} | {'误差(eV)':<10}")
    print("-" * 65)

    for i, idx in enumerate(worst_idx):
        print(f"{i+1:<8} | {idx:<10} | {all_reals[idx]:<12.4f} | {all_preds[idx]:<12.4f} | {errors[idx]:<10.4f}")


if __name__ == "__main__":
    test_on_full_dataset()
