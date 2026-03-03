from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from mainline.baseline.third_party.cgcnn.cgcnn.data import CIFData, collate_pool
from mainline.baseline.third_party.cgcnn.cgcnn.model import CrystalGraphConvNet


ROOT = Path(__file__).resolve().parents[2]


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class Normalizer:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor: torch.Tensor) -> torch.Tensor:
        return normed_tensor * self.std + self.mean

    def state_dict(self) -> dict[str, float]:
        return {"mean": float(self.mean), "std": float(self.std)}

    def load_state_dict(self, state_dict: dict[str, float]) -> None:
        self.mean = torch.tensor(state_dict["mean"])
        self.std = torch.tensor(state_dict["std"])


def mae(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(target - prediction)).item()


def class_eval(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float, float, float]:
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = target.squeeze()
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average="binary"
        )
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


def _load_split_ids(split_dir: Path) -> tuple[list[str], list[str], list[str]]:
    with (split_dir / "train_ids.json").open("r", encoding="utf-8") as f:
        train_ids = [str(x) for x in json.load(f)]
    with (split_dir / "val_ids.json").open("r", encoding="utf-8") as f:
        val_ids = [str(x) for x in json.load(f)]
    with (split_dir / "test_ids.json").open("r", encoding="utf-8") as f:
        test_ids = [str(x) for x in json.load(f)]
    return train_ids, val_ids, test_ids


def _load_id_mapping(mapping_path: Path) -> dict[str, str]:
    with mapping_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    id_map = payload.get("id_map")
    if not isinstance(id_map, dict):
        raise ValueError(f"id_map missing in {mapping_path}")
    return {str(k): str(v) for k, v in id_map.items()}


def _dataset_index_map(dataset: CIFData) -> dict[str, int]:
    out: dict[str, int] = {}
    for idx, row in enumerate(dataset.id_prop_data):
        if len(row) < 1:
            continue
        out[str(row[0])] = idx
    return out


def _make_loader(
    dataset: CIFData,
    indices: list[int],
    batch_size: int,
    workers: int,
    pin_memory: bool,
    shuffle: bool,
) -> DataLoader:
    if shuffle:
        sampler = SubsetRandomSampler(indices)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=workers,
            collate_fn=collate_pool,
            pin_memory=pin_memory,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(indices),
        num_workers=workers,
        collate_fn=collate_pool,
        pin_memory=pin_memory,
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(
    loader: DataLoader,
    model: CrystalGraphConvNet,
    criterion: nn.Module,
    normalizer: Normalizer,
    optimizer: optim.Optimizer | None,
    cuda: bool,
    task: str,
) -> tuple[float, float]:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    losses = AverageMeter()
    metric = AverageMeter()

    for input_data, target, _ in loader:
        if cuda:
            input_var = (
                input_data[0].cuda(non_blocking=True),
                input_data[1].cuda(non_blocking=True),
                input_data[2].cuda(non_blocking=True),
                [idx.cuda(non_blocking=True) for idx in input_data[3]],
            )
        else:
            input_var = input_data

        if task == "regression":
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        target_var = target_normed.cuda(non_blocking=True) if cuda else target_normed

        with torch.set_grad_enabled(is_train):
            output = model(*input_var)
            loss = criterion(output, target_var)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if task == "regression":
            value = mae(normalizer.denorm(output.detach().cpu()), target)
        else:
            value = class_eval(output.detach().cpu(), target)[0]
        losses.update(float(loss.detach().cpu().item()), target.size(0))
        metric.update(float(value), target.size(0))

    return losses.avg, metric.avg


def _eval_and_dump(
    loader: DataLoader,
    model: CrystalGraphConvNet,
    criterion: nn.Module,
    normalizer: Normalizer,
    cuda: bool,
    task: str,
    pred_csv: Path | None,
) -> dict[str, float]:
    model.eval()
    losses = AverageMeter()
    metric = AverageMeter()
    rows: list[tuple[str, float, float]] = []

    for input_data, target, batch_ids in loader:
        if cuda:
            input_var = (
                input_data[0].cuda(non_blocking=True),
                input_data[1].cuda(non_blocking=True),
                input_data[2].cuda(non_blocking=True),
                [idx.cuda(non_blocking=True) for idx in input_data[3]],
            )
        else:
            input_var = input_data

        if task == "regression":
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        target_var = target_normed.cuda(non_blocking=True) if cuda else target_normed

        with torch.no_grad():
            output = model(*input_var)
            loss = criterion(output, target_var)

        if task == "regression":
            preds = normalizer.denorm(output.detach().cpu()).view(-1).tolist()
            reals = target.view(-1).tolist()
            for cid, real, pred in zip(batch_ids, reals, preds):
                rows.append((str(cid), float(real), float(pred)))
            value = mae(normalizer.denorm(output.detach().cpu()), target)
        else:
            value = class_eval(output.detach().cpu(), target)[0]
        losses.update(float(loss.detach().cpu().item()), target.size(0))
        metric.update(float(value), target.size(0))

    if pred_csv is not None and rows:
        pred_csv.parent.mkdir(parents=True, exist_ok=True)
        with pred_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "target", "prediction"])
            writer.writerows(rows)
    if task == "regression":
        out = {"loss": float(losses.avg), "mae": float(metric.avg)}
    else:
        out = {"loss": float(losses.avg), "accuracy": float(metric.avg)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--mapping_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["SGD", "Adam"], default="Adam")
    parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    parser.add_argument("--atom_fea_len", type=int, default=64)
    parser.add_argument("--h_fea_len", type=int, default=128)
    parser.add_argument("--n_conv", type=int, default=3)
    parser.add_argument("--n_h", type=int, default=1)
    parser.add_argument("--radius", type=float, default=8.0)
    parser.add_argument("--max_num_nbr", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--disable_cuda", action="store_true")
    args = parser.parse_args()

    data_root = _resolve(args.data_root)
    split_dir = _resolve(args.split_dir)
    mapping_json = _resolve(args.mapping_json)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cuda = (not args.disable_cuda) and torch.cuda.is_available()
    _seed_everything(args.seed)

    dataset = CIFData(
        str(data_root),
        max_num_nbr=args.max_num_nbr,
        radius=args.radius,
        random_seed=args.seed,
    )

    id_map = _load_id_mapping(mapping_json)
    train_ids, val_ids, test_ids = _load_split_ids(split_dir)
    mapped_train = [id_map[x] for x in train_ids]
    mapped_val = [id_map[x] for x in val_ids]
    mapped_test = [id_map[x] for x in test_ids]

    idx_map = _dataset_index_map(dataset)
    train_idx = [idx_map[x] for x in mapped_train if x in idx_map]
    val_idx = [idx_map[x] for x in mapped_val if x in idx_map]
    test_idx = [idx_map[x] for x in mapped_test if x in idx_map]
    if not train_idx or not val_idx or not test_idx:
        raise ValueError("Mapped split indices are empty; check mapping and dataset.")

    train_loader = _make_loader(dataset, train_idx, args.batch_size, args.workers, cuda, True)
    val_loader = _make_loader(dataset, val_idx, args.batch_size, args.workers, cuda, False)
    test_loader = _make_loader(dataset, test_idx, args.batch_size, args.workers, cuda, False)

    sample_data = [dataset[i] for i in train_idx[: min(len(train_idx), 500)]]
    _, sample_target, _ = collate_pool(sample_data)
    normalizer = Normalizer(sample_target)

    structures, _, _ = dataset[train_idx[0]]
    model = CrystalGraphConvNet(
        structures[0].shape[-1],
        structures[1].shape[-1],
        atom_fea_len=args.atom_fea_len,
        n_conv=args.n_conv,
        h_fea_len=args.h_fea_len,
        n_h=args.n_h,
        classification=(args.task == "classification"),
    )
    if cuda:
        model.cuda()

    criterion = nn.NLLLoss() if args.task == "classification" else nn.MSELoss()
    if args.optimizer == "SGD":
        optimizer: optim.Optimizer = optim.SGD(
            model.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
    else:
        optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    ckpt_path = output_dir / "model_best.pth.tar"
    latest_path = output_dir / "model_latest.pth.tar"

    if args.mode == "train":
        best = 1e10 if args.task == "regression" else 0.0
        for epoch in range(args.epochs):
            train_loss, train_metric = _run_epoch(
                train_loader, model, criterion, normalizer, optimizer, cuda, args.task
            )
            val_loss, val_metric = _run_epoch(
                val_loader, model, criterion, normalizer, None, cuda, args.task
            )
            scheduler.step()

            state = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_mae_error": float(val_metric),
                "optimizer": optimizer.state_dict(),
                "normalizer": normalizer.state_dict(),
                "args": vars(args),
            }
            torch.save(state, latest_path)

            improved = val_metric < best if args.task == "regression" else val_metric > best
            if improved:
                best = val_metric
                torch.save(state, ckpt_path)

            print(
                f"[EPOCH {epoch + 1}/{args.epochs}] "
                f"train_loss={train_loss:.4f} train_metric={train_metric:.4f} "
                f"val_loss={val_loss:.4f} val_metric={val_metric:.4f}"
            )

        if not ckpt_path.exists():
            torch.save(torch.load(latest_path, map_location="cpu"), ckpt_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    normalizer.load_state_dict(checkpoint["normalizer"])
    if cuda:
        model.cuda()

    pred_csv = output_dir / "test_predictions.csv"
    test_metrics = _eval_and_dump(
        test_loader,
        model,
        criterion,
        normalizer,
        cuda,
        args.task,
        pred_csv=pred_csv,
    )
    _write = {
        "mode": args.mode,
        "task": args.task,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "checkpoint": str(ckpt_path),
        **test_metrics,
    }
    metrics_path = output_dir / "metrics_test.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(_write, f, ensure_ascii=False, indent=2)
    print(f"[OK] test metrics: {test_metrics}")
    print(f"[OK] metrics file: {metrics_path}")


if __name__ == "__main__":
    main()
