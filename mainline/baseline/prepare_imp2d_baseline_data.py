from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import torch
try:
    from pymatgen.core import Lattice, Structure
except ModuleNotFoundError:
    Lattice = None
    Structure = None

try:
    from ase import Atoms
    from ase.io import write as ase_write
except ModuleNotFoundError:
    Atoms = None
    ase_write = None


ROOT = Path(__file__).resolve().parents[2]


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def _load_json_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [str(x) for x in json.load(f)]


def _safe_id(value: str) -> str:
    blocked = "\\/:*?\"<>|"
    out = value
    for ch in blocked:
        out = out.replace(ch, "_")
    return out.strip() or "sample"


def _write_csv(path: Path, rows: list[tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _build_atom_init(atom_features_path: Path) -> dict[str, list[float]]:
    features = torch.load(atom_features_path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(features):
        raise TypeError(f"atom_features should be tensor, got {type(features)}")
    if features.ndim != 2:
        raise ValueError(f"atom_features should be 2D, got shape={tuple(features.shape)}")
    out: dict[str, list[float]] = {}
    max_z = min(int(features.shape[0]) - 1, 100)
    for z in range(1, max_z + 1):
        out[str(z)] = features[z].to(torch.float32).tolist()
    return out


def _write_cif(cell, numbers, coords, out_path: Path) -> None:
    if Structure is not None and Lattice is not None:
        struct = Structure(
            lattice=Lattice(cell),
            species=numbers,
            coords=coords,
            coords_are_cartesian=True,
        )
        struct.to(filename=str(out_path), fmt="cif")
        return
    if Atoms is not None and ase_write is not None:
        atoms = Atoms(numbers=numbers, positions=coords, cell=cell, pbc=True)
        ase_write(str(out_path), atoms, format="cif")
        return
    raise ModuleNotFoundError(
        "Need one of these packages to write CIF: pymatgen or ase."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="WLY/final_dataset.pkl")
    parser.add_argument("--atom_features", default="WLY/atom_features.pth")
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--split_version", default="v1")
    parser.add_argument("--output_root", default="mainline/baseline/datasets")
    parser.add_argument("--alignn_epochs", type=int, default=300)
    parser.add_argument("--alignn_batch_size", type=int, default=32)
    parser.add_argument("--alignn_lr", type=float, default=0.001)
    parser.add_argument("--alignn_workers", type=int, default=4)
    parser.add_argument("--alignn_cutoff", type=float, default=8.0)
    parser.add_argument("--alignn_max_neighbors", type=int, default=12)
    parser.add_argument("--alignn_output_dir", default="")
    args = parser.parse_args()

    dataset_path = _resolve(args.dataset)
    atom_features_path = _resolve(args.atom_features)
    split_dir = _resolve(args.split_dir)
    output_root = _resolve(args.output_root) / f"imp2d_{args.split_version}"

    with dataset_path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected list dataset at {dataset_path}, got {type(data)}")

    train_ids = _load_json_list(split_dir / "train_ids.json")
    val_ids = _load_json_list(split_dir / "val_ids.json")
    test_ids = _load_json_list(split_dir / "test_ids.json")
    ordered_ids = train_ids + val_ids + test_ids

    id_to_sample: dict[str, dict] = {}
    for idx, sample in enumerate(data):
        raw_id = sample.get("id")
        sid = str(raw_id) if raw_id is not None and str(raw_id) != "" else str(idx)
        if sid not in id_to_sample:
            id_to_sample[sid] = sample

    missing = [sid for sid in ordered_ids if sid not in id_to_sample]
    if missing:
        raise KeyError(f"IDs from split not found in dataset, first 5: {missing[:5]}")

    cgcnn_root = output_root / "cgcnn"
    alignn_root = output_root / "alignn"
    cgcnn_struct_root = cgcnn_root / "structures"
    alignn_struct_root = alignn_root / "structures"
    cgcnn_struct_root.mkdir(parents=True, exist_ok=True)
    alignn_struct_root.mkdir(parents=True, exist_ok=True)

    mapped_ids: dict[str, str] = {}
    seen: set[str] = set()
    for sid in ordered_ids:
        file_id = _safe_id(sid)
        suffix = 1
        while file_id in seen:
            suffix += 1
            file_id = f"{_safe_id(sid)}_{suffix}"
        seen.add(file_id)
        mapped_ids[sid] = file_id

    cgcnn_rows: list[tuple[str, float]] = []
    alignn_rows: list[tuple[str, float]] = []
    for sid in ordered_ids:
        sample = id_to_sample[sid]
        fid = mapped_ids[sid]
        target = float(sample["target"])
        numbers = [int(x) for x in sample["numbers"]]
        coords = sample["positions"]
        cell = sample["cell"]
        cgcnn_cif = cgcnn_struct_root / f"{fid}.cif"
        alignn_cif = alignn_struct_root / f"{fid}.cif"
        _write_cif(cell=cell, numbers=numbers, coords=coords, out_path=cgcnn_cif)
        _write_cif(cell=cell, numbers=numbers, coords=coords, out_path=alignn_cif)

        cgcnn_rows.append((f"structures/{fid}", target))
        alignn_rows.append((f"structures/{fid}.cif", target))

    _write_csv(cgcnn_root / "id_prop.csv", cgcnn_rows)
    atom_init = _build_atom_init(atom_features_path)
    _write_json(cgcnn_root / "atom_init.json", atom_init)

    _write_csv(alignn_root / "id_prop.csv", alignn_rows)
    alignn_output_dir = (
        _resolve(args.alignn_output_dir)
        if args.alignn_output_dir
        else ROOT / "mainline" / "baseline" / "runs" / f"alignn_{args.split_version}"
    )
    alignn_config = {
        "dataset": "user_data",
        "target": "target",
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "epochs": int(args.alignn_epochs),
        "batch_size": int(args.alignn_batch_size),
        "weight_decay": 1e-5,
        "learning_rate": float(args.alignn_lr),
        "criterion": "mse",
        "optimizer": "adamw",
        "scheduler": "onecycle",
        "pin_memory": False,
        "save_dataloader": False,
        "write_predictions": True,
        "store_outputs": True,
        "progress": True,
        "log_tensorboard": False,
        "standard_scalar_and_pca": False,
        "use_canonize": True,
        "compute_line_graph": True,
        "num_workers": int(args.alignn_workers),
        "cutoff": float(args.alignn_cutoff),
        "cutoff_extra": 3.0,
        "max_neighbors": int(args.alignn_max_neighbors),
        "keep_data_order": True,
        "output_dir": str(alignn_output_dir),
        "use_lmdb": False,
        "model": {
            "name": "alignn",
        },
    }
    _write_json(alignn_root / "config_imp2d.json", alignn_config)
    _write_json(
        output_root / "mapping.json",
        {
            "split_version": args.split_version,
            "n_total": len(ordered_ids),
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "n_test": len(test_ids),
            "id_map": mapped_ids,
        },
    )
    print(f"[OK] Prepared baseline data: {output_root}")
    print(f"[OK] CGCNN root: {cgcnn_root}")
    print(f"[OK] ALIGNN root: {alignn_root}")


if __name__ == "__main__":
    main()
