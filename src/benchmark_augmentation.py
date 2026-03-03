import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from data.data_loader import CrystalGraphDataset, collate_fn
except ImportError:
    from WLY.data_loader import CrystalGraphDataset, collate_fn

try:
    from model import CrystalTransformer
except ImportError:
    from src.model import CrystalTransformer


def move_to_device(obj, device, non_blocking=False):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device, non_blocking=non_blocking) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [move_to_device(v, device, non_blocking=non_blocking) for v in obj]
        return type(obj)(converted)
    return obj


def get_base_id(uid):
    if "_rot_" in uid:
        return uid.split("_rot_")[0]
    if "_pert_" in uid:
        return uid.split("_pert_")[0]
    return uid


def is_original(sample, id_key):
    if "metadata" in sample and "augmented" in sample["metadata"]:
        return False
    uid = str(sample.get(id_key, ""))
    if "_rot_" in uid or "_pert_" in uid:
        return False
    return True


def detect_id_key(samples):
    candidates = ("unique_id", "id", "uid", "mp_id")
    for key in candidates:
        if all(key in s for s in samples[: min(64, len(samples))]):
            return key
    raise KeyError("Cannot find ID key in dataset samples.")


def load_split_ids(split_dir):
    with open(os.path.join(split_dir, "train_ids.json"), "r", encoding="utf-8") as f:
        train_ids = [str(x) for x in json.load(f)]
    with open(os.path.join(split_dir, "val_ids.json"), "r", encoding="utf-8") as f:
        val_ids = [str(x) for x in json.load(f)]
    with open(os.path.join(split_dir, "test_ids.json"), "r", encoding="utf-8") as f:
        test_ids = [str(x) for x in json.load(f)]
    return train_ids, val_ids, test_ids


def make_id_to_index(dataset, id_key):
    out = {}
    for i, sample in enumerate(dataset.data):
        sid = str(sample.get(id_key, i))
        if sid not in out:
            out[sid] = i
    return out


def split_from_locked_ids(dataset, id_key, train_ids, val_ids, test_ids):
    id_to_index = make_id_to_index(dataset, id_key)
    train_indices = [id_to_index[s] for s in train_ids if s in id_to_index]
    val_indices = [id_to_index[s] for s in val_ids if s in id_to_index]
    test_indices = [id_to_index[s] for s in test_ids if s in id_to_index]
    if not train_indices or not val_indices or not test_indices:
        raise ValueError("Locked split IDs cannot map to dataset indices.")
    return train_indices, val_indices, test_indices


def split_grouped(dataset, id_key, train_ratio, val_ratio, seed):
    groups = defaultdict(list)
    for idx, sample in enumerate(dataset.data):
        uid = str(sample[id_key])
        base_id = get_base_id(uid)
        groups[base_id].append(idx)

    base_ids = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(base_ids)

    n_total = len(base_ids)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    train_ids = base_ids[:n_train]
    val_ids = base_ids[n_train : n_train + n_val]
    test_ids = base_ids[n_train + n_val :]

    train_indices = []
    val_indices = []
    test_indices = []

    for uid in train_ids:
        train_indices.extend(groups[uid])
    for uid in val_ids:
        for idx in groups[uid]:
            if is_original(dataset.data[idx], id_key):
                val_indices.append(idx)
    for uid in test_ids:
        for idx in groups[uid]:
            if is_original(dataset.data[idx], id_key):
                test_indices.append(idx)

    if not train_indices or not val_indices or not test_indices:
        raise ValueError("Grouped split produced empty subset.")
    return train_indices, val_indices, test_indices


def split_random(dataset, train_ratio, val_ratio, seed):
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=gen)
    return list(train_set.indices), list(val_set.indices), list(test_set.indices)


def indices_to_ids(dataset, indices, id_key):
    return [str(dataset.data[i].get(id_key, i)) for i in indices]


def create_model(checkpoint, device, fallback_hidden, fallback_local, fallback_global):
    cfg = checkpoint.get("config", {})
    hidden_dim = int(cfg.get("hidden_dim", fallback_hidden))
    n_local = int(cfg.get("n_local", fallback_local))
    n_global = int(cfg.get("n_global", fallback_global))
    model = CrystalTransformer(
        atom_fea_len=9,
        hidden_dim=hidden_dim,
        n_local_layers=n_local,
        n_global_layers=n_global,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, hidden_dim, n_local, n_global


def evaluate_subset(model, loader, device, norm_mean, norm_std):
    all_preds = []
    all_reals = []
    all_ids = []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            batch = move_to_device(batch, device, non_blocking=(device.type == "cuda"))
            preds_norm = model(batch)
            targets = batch["target"]
            preds = preds_norm * norm_std + norm_mean
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_reals.extend(targets.detach().cpu().numpy().tolist())
            if "ids" in batch:
                all_ids.extend([str(x) for x in batch["ids"]])
            else:
                all_ids.extend([""] * len(targets))
    preds = np.array(all_preds, dtype=np.float64)
    reals = np.array(all_reals, dtype=np.float64)
    abs_err = np.abs(preds - reals)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((preds - reals) ** 2)))
    ss_res = float(np.sum((preds - reals) ** 2))
    ss_tot = float(np.sum((reals - np.mean(reals)) ** 2))
    r2 = float("nan") if ss_tot <= 0 else 1.0 - (ss_res / ss_tot)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "n": int(len(reals)),
        "ids": all_ids,
        "preds": preds,
        "reals": reals,
        "errors": abs_err,
    }


def save_predictions(path, ids, preds, reals):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "target", "prediction", "abs_error"])
        for sid, t, p in zip(ids, reals, preds):
            writer.writerow([sid, float(t), float(p), float(abs(t - p))])


class _SubsetWithIds:
    def __init__(self, base_dataset, indices, id_key):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.id_key = id_key

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        item = self.base_dataset[real_idx]
        if isinstance(item, dict):
            item = dict(item)
            sample = self.base_dataset.data[real_idx]
            item["ids"] = str(sample.get(self.id_key, real_idx))
            return item
        return item


def _collate_dict_with_ids(batch):
    first = batch[0]
    ids = [x["ids"] for x in batch]
    stripped = []
    for x in batch:
        y = dict(x)
        del y["ids"]
        stripped.append(y)
    packed = collate_fn(stripped)
    packed["ids"] = ids
    return packed


def _extract_id_token(raw_id):
    sid = str(raw_id).strip()
    sid = sid.replace("\\", "/")
    if "/" in sid:
        sid = sid.split("/")[-1]
    if sid.endswith(".cif"):
        sid = sid[:-4]
    return sid


def _load_mapping(mapping_json):
    with open(mapping_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    id_map = data.get("id_map", {})
    reverse = {str(v): str(k) for k, v in id_map.items()}
    return reverse


def evaluate_external_predictions(pred_csv, reverse_mapping, allowed_ids=None):
    preds = []
    reals = []
    ids = []
    with open(pred_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = row.get("id")
            if row_id is None:
                continue
            token = _extract_id_token(row_id)
            orig_id = reverse_mapping.get(token, token)
            if allowed_ids is not None and orig_id not in allowed_ids:
                continue
            target = row.get("target")
            prediction = row.get("prediction")
            if target is None or prediction is None:
                continue
            try:
                t = float(target)
                p = float(prediction)
            except ValueError:
                continue
            ids.append(orig_id)
            reals.append(t)
            preds.append(p)
    if len(preds) == 0:
        raise ValueError(f"No valid rows found in external prediction file: {pred_csv}")
    preds = np.array(preds, dtype=np.float64)
    reals = np.array(reals, dtype=np.float64)
    abs_err = np.abs(preds - reals)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((preds - reals) ** 2)))
    ss_res = float(np.sum((preds - reals) ** 2))
    ss_tot = float(np.sum((reals - np.mean(reals)) ** 2))
    r2 = float("nan") if ss_tot <= 0 else 1.0 - (ss_res / ss_tot)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "n": int(len(reals)),
        "ids": ids,
        "preds": preds,
        "reals": reals,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--checkpoints", nargs="+", default=[])
    parser.add_argument("--names", nargs="+", default=[])
    parser.add_argument("--cgcnn_pred_csv", default="")
    parser.add_argument("--alignn_pred_csv", default="")
    parser.add_argument("--mapping_json", default="")
    parser.add_argument("--split_dir", default="")
    parser.add_argument("--output_dir", default="benchmark_outputs")
    parser.add_argument("--split_mode", choices=["grouped", "random"], default="grouped")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_local", type=int, default=2)
    parser.add_argument("--n_global", type=int, default=1)
    args = parser.parse_args()

    if len(args.checkpoints) == 0 and not args.cgcnn_pred_csv and not args.alignn_pred_csv:
        raise ValueError("Provide at least one of: --checkpoints, --cgcnn_pred_csv, --alignn_pred_csv")
    if args.names and len(args.names) != len(args.checkpoints):
        raise ValueError("--names length must equal --checkpoints length")
    if (args.cgcnn_pred_csv or args.alignn_pred_csv) and not args.mapping_json:
        raise ValueError("--mapping_json is required when using external prediction CSVs")

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = CrystalGraphDataset(args.data, args.features, device="cpu")
    id_key = detect_id_key(dataset.data)
    if args.split_dir:
        train_ids, val_ids, test_ids = load_split_ids(args.split_dir)
        train_indices, val_indices, test_indices = split_from_locked_ids(
            dataset, id_key=id_key, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids
        )
    else:
        if args.split_mode == "grouped":
            train_indices, val_indices, test_indices = split_grouped(
                dataset, id_key=id_key, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
            )
        else:
            train_indices, val_indices, test_indices = split_random(
                dataset, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
            )

    val_subset = _SubsetWithIds(dataset, val_indices, id_key)
    test_subset = _SubsetWithIds(dataset, test_indices, id_key)

    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_dict_with_ids,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_dict_with_ids,
    )

    test_id_set = set(indices_to_ids(dataset, test_indices, id_key))

    os.makedirs(args.output_dir, exist_ok=True)
    split_out = {
        "split_mode": args.split_mode,
        "id_key": id_key,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "n_test": len(test_indices),
        "train_ids": indices_to_ids(dataset, train_indices, id_key),
        "val_ids": indices_to_ids(dataset, val_indices, id_key),
        "test_ids": indices_to_ids(dataset, test_indices, id_key),
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
    }
    with open(os.path.join(args.output_dir, "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(split_out, f, ensure_ascii=False, indent=2)

    summary_rows = []
    for i, ckpt_path in enumerate(args.checkpoints):
        name = args.names[i] if args.names else os.path.splitext(os.path.basename(ckpt_path))[0]
        checkpoint = torch.load(ckpt_path, map_location=device)
        model, hidden_dim, n_local, n_global = create_model(
            checkpoint,
            device,
            fallback_hidden=args.hidden_dim,
            fallback_local=args.n_local,
            fallback_global=args.n_global,
        )
        norm_mean = checkpoint["normalizer"]["mean"]
        norm_std = checkpoint["normalizer"]["std"]
        val_metrics = evaluate_subset(model, val_loader, device, norm_mean, norm_std)
        test_metrics = evaluate_subset(model, test_loader, device, norm_mean, norm_std)

        save_predictions(
            os.path.join(args.output_dir, f"{name}_val_predictions.csv"),
            val_metrics["ids"],
            val_metrics["preds"],
            val_metrics["reals"],
        )
        save_predictions(
            os.path.join(args.output_dir, f"{name}_test_predictions.csv"),
            test_metrics["ids"],
            test_metrics["preds"],
            test_metrics["reals"],
        )

        item = {
            "name": name,
            "checkpoint": ckpt_path,
            "device": str(device),
            "hidden_dim": hidden_dim,
            "n_local": n_local,
            "n_global": n_global,
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "val_n": val_metrics["n"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
            "test_n": test_metrics["n"],
        }
        summary_rows.append(item)
        with open(os.path.join(args.output_dir, f"{name}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(item, f, ensure_ascii=False, indent=2)

    reverse_mapping = _load_mapping(args.mapping_json) if args.mapping_json else {}

    if args.cgcnn_pred_csv:
        cgcnn_metrics = evaluate_external_predictions(
            args.cgcnn_pred_csv,
            reverse_mapping=reverse_mapping,
            allowed_ids=test_id_set,
        )
        save_predictions(
            os.path.join(args.output_dir, "cgcnn_test_predictions.csv"),
            cgcnn_metrics["ids"],
            cgcnn_metrics["preds"],
            cgcnn_metrics["reals"],
        )
        item = {
            "name": "cgcnn",
            "checkpoint": args.cgcnn_pred_csv,
            "device": "external",
            "hidden_dim": None,
            "n_local": None,
            "n_global": None,
            "val_mae": None,
            "val_rmse": None,
            "val_r2": None,
            "val_n": None,
            "test_mae": cgcnn_metrics["mae"],
            "test_rmse": cgcnn_metrics["rmse"],
            "test_r2": cgcnn_metrics["r2"],
            "test_n": cgcnn_metrics["n"],
        }
        summary_rows.append(item)
        with open(os.path.join(args.output_dir, "cgcnn_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(item, f, ensure_ascii=False, indent=2)

    if args.alignn_pred_csv:
        alignn_metrics = evaluate_external_predictions(
            args.alignn_pred_csv,
            reverse_mapping=reverse_mapping,
            allowed_ids=test_id_set,
        )
        save_predictions(
            os.path.join(args.output_dir, "alignn_test_predictions.csv"),
            alignn_metrics["ids"],
            alignn_metrics["preds"],
            alignn_metrics["reals"],
        )
        item = {
            "name": "alignn",
            "checkpoint": args.alignn_pred_csv,
            "device": "external",
            "hidden_dim": None,
            "n_local": None,
            "n_global": None,
            "val_mae": None,
            "val_rmse": None,
            "val_r2": None,
            "val_n": None,
            "test_mae": alignn_metrics["mae"],
            "test_rmse": alignn_metrics["rmse"],
            "test_r2": alignn_metrics["r2"],
            "test_n": alignn_metrics["n"],
        }
        summary_rows.append(item)
        with open(os.path.join(args.output_dir, "alignn_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(item, f, ensure_ascii=False, indent=2)

    if len(summary_rows) == 0:
        raise ValueError("No benchmark results generated.")
    summary_rows.sort(key=lambda x: x["test_mae"])
    with open(os.path.join(args.output_dir, "benchmark_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "benchmark_summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Benchmark finished.")
    for row in summary_rows:
        print(
            f"{row['name']}: test_mae={row['test_mae']:.4f}, "
            f"test_rmse={row['test_rmse']:.4f}, test_r2={row['test_r2']:.4f}"
        )


if __name__ == "__main__":
    main()
