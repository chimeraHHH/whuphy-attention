from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"[RUN] cwd={cwd} :: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _prepare(split_dir: Path, split_version: str, alignn_epochs: int) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "mainline" / "baseline" / "prepare_imp2d_baseline_data.py"),
        "--dataset",
        str(ROOT / "WLY" / "final_dataset.pkl"),
        "--atom_features",
        str(ROOT / "WLY" / "atom_features.pth"),
        "--split_dir",
        str(split_dir),
        "--split_version",
        split_version,
        "--alignn_epochs",
        str(int(alignn_epochs)),
    ]
    _run(cmd, ROOT)


def _run_cgcnn(action: str, split_dir: Path, split_version: str, cgcnn_epochs: int) -> None:
    output_dir = ROOT / "mainline" / "baseline" / "runs" / f"cgcnn_{split_version}"
    cmd = [
        sys.executable,
        str(ROOT / "mainline" / "baseline" / "cgcnn_locked.py"),
        "--data_root",
        str(ROOT / "mainline" / "baseline" / "datasets" / f"imp2d_{split_version}" / "cgcnn"),
        "--split_dir",
        str(split_dir),
        "--mapping_json",
        str(ROOT / "mainline" / "baseline" / "datasets" / f"imp2d_{split_version}" / "mapping.json"),
        "--output_dir",
        str(output_dir),
        "--mode",
        "train" if action == "train" else "eval",
        "--batch_size",
        "64",
        "--workers",
        "4",
        "--lr",
        "0.001",
        "--optimizer",
        "Adam",
        "--epochs",
        str(int(cgcnn_epochs)),
        "--seed",
        "123",
        "--radius",
        "10.0",
        "--max_num_nbr",
        "16",
    ]
    _run(cmd, ROOT)


def _run_alignn_train(split_version: str) -> None:
    alignn_workdir = ROOT / "mainline" / "baseline" / "third_party" / "alignn"
    cmd = [
        sys.executable,
        "-m",
        "alignn.train_alignn",
        "--root_dir",
        str(ROOT / "mainline" / "baseline" / "datasets" / f"imp2d_{split_version}" / "alignn"),
        "--config_name",
        str(
            ROOT
            / "mainline"
            / "baseline"
            / "datasets"
            / f"imp2d_{split_version}"
            / "alignn"
            / "config_imp2d.json"
        ),
        "--file_format",
        "cif",
        "--output_dir",
        str(ROOT / "mainline" / "baseline" / "runs" / f"alignn_{split_version}"),
    ]
    _run(cmd, alignn_workdir)


def _run_alignn_eval(split_version: str) -> None:
    run_dir = ROOT / "mainline" / "baseline" / "runs" / f"alignn_{split_version}"
    pred_file = run_dir / "prediction_results_test_set.csv"
    if not pred_file.exists():
        raise FileNotFoundError(
            f"ALIGNN test prediction file not found: {pred_file}. Run train first."
        )
    print(f"[OK] ALIGNN test predictions ready: {pred_file}")
    summary = {
        "split_version": split_version,
        "prediction_file": str(pred_file),
    }
    out_file = run_dir / "eval_summary.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] ALIGNN eval summary: {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cgcnn", "alignn"], required=True)
    parser.add_argument("--action", choices=["train", "eval"], required=True)
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--split_version", required=True)
    parser.add_argument("--cgcnn_epochs", type=int, default=200)
    parser.add_argument("--alignn_epochs", type=int, default=300)
    args = parser.parse_args()

    split_dir = _resolve(args.split_dir)

    if args.model == "cgcnn":
        if args.action == "train":
            _prepare(split_dir, args.split_version, args.alignn_epochs)
        _run_cgcnn(args.action, split_dir, args.split_version, args.cgcnn_epochs)
        return
    if args.model == "alignn":
        if args.action == "train":
            _prepare(split_dir, args.split_version, args.alignn_epochs)
            _run_alignn_train(args.split_version)
        else:
            _run_alignn_eval(args.split_version)
        return

    raise ValueError(f"Unsupported model: {args.model}")


if __name__ == "__main__":
    main()
