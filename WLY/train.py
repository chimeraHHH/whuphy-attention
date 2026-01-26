import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
import os
import time
import numpy as np
from tqdm import tqdm
import gc
import argparse

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

def move_to_device(obj, device, non_blocking=False):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device, non_blocking=non_blocking) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [move_to_device(v, device, non_blocking=non_blocking) for v in obj]
        return type(obj)(converted)
    return obj

def is_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return world_size > 1

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(base_dir, "final_dataset.pkl"))
    parser.add_argument("--features", default=os.path.join(base_dir, "atom_features.pth"))
    parser.add_argument("--output_dir", default=os.path.join(base_dir, "checkpoints"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_local", type=int, default=2)
    parser.add_argument("--n_global", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--restart_each_epoch", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    default_resume = os.path.join(output_dir, "latest_model.pth")
    resume_path = None
    if not args.no_resume:
        resume_path = args.resume or default_resume

    CONFIG = {
        "data_path": args.data,
        "feature_path": args.features,
        "output_dir": output_dir,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "n_local": args.n_local,
        "n_global": args.n_global,
        "seed": args.seed,
        "resume_path": resume_path,
        "restart_each_epoch": args.restart_each_epoch,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers,
        "prefetch_factor": args.prefetch_factor,
        "backend": args.backend,
        "distributed": args.distributed,
        "fp16": args.fp16,
    }
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    distributed = CONFIG["distributed"] or is_distributed()
    rank = 0
    local_rank = 0
    world_size = 1
    if distributed:
        if torch.cuda.is_available():
            backend = CONFIG["backend"] or "nccl"
        else:
            backend = CONFIG["backend"] or "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    is_main = (rank == 0)
    if is_main:
        print(f"Using device: {device} | distributed={distributed} world_size={world_size}")

    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG["seed"])

    # --- 1. Load Data ---
    if is_main:
        print("Loading dataset...")
    full_dataset = CrystalGraphDataset(CONFIG["data_path"], CONFIG["feature_path"], device="cpu")
    
    all_targets = [sample['target'] for sample in full_dataset.data]
    target_tensor = torch.tensor(all_targets, dtype=torch.float32, device=device)
    normalizer = Normalizer(target_tensor)
    if is_main:
        print(f"Target Norm Stats: Mean={normalizer.mean:.4f}, Std={normalizer.std:.4f}")
    
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    gen = torch.Generator().manual_seed(CONFIG["seed"])
    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size], generator=gen)

    pin_memory = bool(CONFIG["pin_memory"]) or (device.type == "cuda")
    persistent_workers = bool(CONFIG["persistent_workers"]) and CONFIG["num_workers"] > 0

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=CONFIG["seed"]) if distributed else None
    train_loader = DataLoader(
        train_set,
        batch_size=CONFIG["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=CONFIG["num_workers"],
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=(CONFIG["prefetch_factor"] if CONFIG["num_workers"] > 0 else None),
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=(CONFIG["prefetch_factor"] if CONFIG["num_workers"] > 0 else None),
        collate_fn=collate_fn,
        drop_last=False,
    )
    
    # --- 2. Initialize Model & Optimizer ---
    model = CrystalTransformer(
        atom_fea_len=9,
        hidden_dim=CONFIG['hidden_dim'],
        n_local_layers=CONFIG['n_local'],
        n_global_layers=CONFIG['n_global']
    ).to(device)

    if distributed and device.type == "cuda":
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    elif distributed:
        model = DDP(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()
    amp_enabled = bool(CONFIG["fp16"]) and (device.type == "cuda")
    use_bf16 = amp_enabled and torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and (not use_bf16)))
    
    start_epoch = 0
    best_val_mae = float('inf')

    # --- 3. Resume Training (断点续传逻辑) ---
    if CONFIG['resume_path'] and os.path.exists(CONFIG['resume_path']):
        if is_main:
            print(f"Resuming from checkpoint: {CONFIG['resume_path']}")
        checkpoint = torch.load(CONFIG['resume_path'], map_location=device)
        state_dict = checkpoint["model_state_dict"]
        if isinstance(model, DDP):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_mae = checkpoint['val_mae']
        if is_main:
            print(f"Restarting from Epoch {start_epoch+1}, Best Val MAE was: {best_val_mae:.4f}")

    if is_main:
        print("\nStart Training...")
    
    for epoch in range(start_epoch, CONFIG['epochs']):
        start_time = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # --- Train ---
        model.train()
        train_loss_sum = 0
        train_mae_sum = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", unit="batch", disable=(not is_main))
        
        for batch in train_pbar:
            optimizer.zero_grad()
            batch = move_to_device(batch, device, non_blocking=pin_memory)
            targets = batch["target"]
            
            targets_norm = normalizer.norm(targets)
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled, dtype=autocast_dtype):
                preds = model(batch)
                loss = criterion(preds, targets_norm)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            batch_loss = loss.item()
            train_loss_sum += batch_loss * targets.size(0)
            with torch.no_grad():
                preds_denorm = normalizer.denorm(preds.float())
                mae = torch.abs(preds_denorm - targets).mean().item()
                train_mae_sum += mae * targets.size(0)
            
            train_pbar.set_postfix({'loss': f'{batch_loss:.4f}', 'mae': f'{mae:.4f}'})

        train_count = torch.tensor(float(len(train_set)), device=device)
        train_loss_sum_t = torch.tensor(train_loss_sum, device=device)
        train_mae_sum_t = torch.tensor(train_mae_sum, device=device)
        if distributed:
            dist.all_reduce(train_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_loss_sum_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_mae_sum_t, op=dist.ReduceOp.SUM)
        avg_train_loss = (train_loss_sum_t / train_count).item()
        avg_train_mae = (train_mae_sum_t / train_count).item()
        
        # --- Validation ---
        model.eval()
        val_loss_sum = 0
        val_mae_sum = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Val]", unit="batch", disable=(not is_main))
        
        with torch.no_grad():
            for batch in val_pbar:
                batch = move_to_device(batch, device, non_blocking=pin_memory)
                targets = batch["target"]
                targets_norm = normalizer.norm(targets)
                with torch.amp.autocast(device_type="cuda", enabled=amp_enabled, dtype=autocast_dtype):
                    preds = model(batch)
                loss = criterion(preds, targets_norm)
                val_loss_sum += loss.item() * targets.size(0)
                
                preds_denorm = normalizer.denorm(preds.float())
                mae = torch.abs(preds_denorm - targets).mean().item()
                val_mae_sum += mae * targets.size(0)

        val_count = torch.tensor(float(len(val_set)), device=device)
        val_loss_sum_t = torch.tensor(val_loss_sum, device=device)
        val_mae_sum_t = torch.tensor(val_mae_sum, device=device)
        if distributed:
            dist.all_reduce(val_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_sum_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_mae_sum_t, op=dist.ReduceOp.SUM)
        avg_val_loss = (val_loss_sum_t / val_count).item()
        avg_val_mae = (val_mae_sum_t / val_count).item()
        
        scheduler.step(avg_val_mae)
        
        saved_msg = ""
        if is_main:
            module = model.module if isinstance(model, DDP) else model
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "normalizer": {"mean": normalizer.mean, "std": normalizer.std},
                "config": CONFIG,
                "val_mae": avg_val_mae,
            }

            latest_path = os.path.join(CONFIG["output_dir"], "latest_model.pth")
            torch.save(checkpoint_data, latest_path)

            if avg_val_mae < best_val_mae:
                best_val_mae = avg_val_mae
                best_path = os.path.join(CONFIG["output_dir"], "best_model.pth")
                torch.save(checkpoint_data, best_path)
                saved_msg = "(*)"
            
        # 强制 Python 垃圾回收
        gc.collect()
            
        epoch_time = time.time() - start_time
        if is_main:
            print(
                f"Epoch {epoch+1}/{CONFIG['epochs']} | "
                f"Train Loss: {avg_train_loss:.4f} MAE: {avg_train_mae:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} MAE: {avg_val_mae:.4f} | "
                f"Time: {epoch_time:.1f}s {saved_msg}"
            )

        if CONFIG.get("restart_each_epoch"):
            if is_main:
                print("Epoch finished, exiting for external restart...")
            break

    if distributed:
        dist.barrier()
        dist.destroy_process_group()

    if is_main:
        print("\nTraining Complete.")
        print(f"Best Validation MAE: {best_val_mae:.4f} eV")

if __name__ == "__main__":
    main()
