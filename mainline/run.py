#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.toml"


def load_config(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print(f"[RUN] cwd={cwd} :: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _as_flag(value: bool, flag: str) -> list[str]:
    return [flag] if value else []


def pull_or_update_repo(
    *,
    repo_url: str,
    dest_dir: Path,
    branch: str | None,
    depth: int,
) -> None:
    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    if dest_dir.exists():
        if not (dest_dir / ".git").exists():
            raise RuntimeError(
                f"Destination exists but is not a git repo: {dest_dir}"
            )
        cmd = ["git", "-C", str(dest_dir), "pull", "--ff-only", "origin"]
        if branch:
            cmd.append(branch)
        run_cmd(cmd, ROOT)
        return

    cmd = ["git", "clone"]
    if depth > 0:
        cmd.extend(["--depth", str(depth)])
    if branch:
        cmd.extend(["--branch", branch])
    cmd.extend([repo_url, str(dest_dir)])
    run_cmd(cmd, ROOT)


def run_pull_baselines(cfg: dict, baseline: str) -> None:
    pull_cfg = cfg["baseline"]["pull"]
    root_dir = ROOT / pull_cfg["root_dir"]
    depth = int(pull_cfg.get("depth", 1))

    if baseline == "all":
        selected = ["cgcnn", "alignn", "ours"]
    else:
        selected = [baseline]

    for name in selected:
        bcfg = pull_cfg[name]
        if not bool(bcfg.get("enabled", True)):
            print(f"[SKIP] baseline '{name}' disabled in config.")
            continue

        if name == "ours":
            ours_path = ROOT / bcfg.get("path", "mainline")
            if not ours_path.exists():
                raise FileNotFoundError(f"Ours path not found: {ours_path}")
            print(f"[OK] Ours baseline path: {ours_path}")
            continue

        repo_url = bcfg["repo"]
        branch = bcfg.get("branch")
        dest_name = bcfg.get("dest", name)
        dest_dir = root_dir / dest_name
        print(f"[BASELINE] {name} -> {dest_dir}")
        pull_or_update_repo(
            repo_url=repo_url,
            dest_dir=dest_dir,
            branch=branch,
            depth=depth,
        )


def run_wly_preprocess(cfg: dict) -> None:
    py = cfg["paths"]["python"]
    wly_dir = ROOT / cfg["paths"]["wly_dir"]
    c = cfg["wly"]["preprocess"]

    cmd = [
        py,
        "pt_to_pickle.py",
        "--input_dir",
        c["input_dir"],
        "--output",
        c["cleaned_output"],
    ]
    if int(c.get("limit", 0)) > 0:
        cmd.extend(["--limit", str(int(c["limit"]))])
    run_cmd(cmd, wly_dir)

    run_cmd(
        [
            py,
            "process_graphs.py",
            "--input",
            c["cleaned_output"],
            "--output",
            c["processed_output"],
            "--cutoff",
            str(c["cutoff"]),
        ],
        wly_dir,
    )

    run_cmd(
        [
            py,
            "filter_dataset.py",
            "--input",
            c["processed_output"],
            "--output",
            c["final_output"],
            "--min",
            str(c["min_target"]),
            "--max",
            str(c["max_target"]),
        ],
        wly_dir,
    )


def run_wly_train(cfg: dict) -> None:
    train_cfg = cfg["wly"]["train"]
    if not bool(train_cfg.get("enabled", True)):
        print("[SKIP] WLY training disabled in config.")
        return

    py = cfg["paths"]["python"]
    wly_dir = ROOT / cfg["paths"]["wly_dir"]

    cmd = [
        py,
        "train.py",
        "--data",
        train_cfg["data"],
        "--features",
        train_cfg["features"],
        "--output_dir",
        train_cfg["output_dir"],
        "--batch_size",
        str(train_cfg["batch_size"]),
        "--epochs",
        str(train_cfg["epochs"]),
        "--lr",
        str(train_cfg["lr"]),
        "--hidden_dim",
        str(train_cfg["hidden_dim"]),
        "--n_local",
        str(train_cfg["n_local"]),
        "--n_global",
        str(train_cfg["n_global"]),
        "--seed",
        str(train_cfg["seed"]),
        "--num_workers",
        str(train_cfg["num_workers"]),
        "--backend",
        train_cfg["backend"],
    ]
    cmd.extend(_as_flag(bool(train_cfg.get("pin_memory", False)), "--pin_memory"))
    cmd.extend(_as_flag(bool(train_cfg.get("fp16", False)), "--fp16"))
    cmd.extend(_as_flag(bool(train_cfg.get("distributed", False)), "--distributed"))
    run_cmd(cmd, wly_dir)


def run_matformer_train(cfg: dict) -> None:
    train_cfg = cfg["matformer"]["train"]
    if not bool(train_cfg.get("enabled", False)):
        print("[SKIP] Matformer training disabled in config.")
        return

    py = cfg["paths"]["python"]
    mat_dir = ROOT / cfg["paths"]["matformer_dir"]

    cmd = [
        py,
        "matformer/scripts/run_edos_pdos.py",
        "--task",
        train_cfg["task"],
        "--epochs",
        str(train_cfg["epochs"]),
        "--batch_size",
        str(train_cfg["batch_size"]),
        "--lr",
        str(train_cfg["lr"]),
        "--cutoff",
        str(train_cfg["cutoff"]),
        "--max_neighbors",
        str(train_cfg["max_neighbors"]),
        "--num_workers",
        str(train_cfg["num_workers"]),
        "--output_root",
        train_cfg["output_root"],
        "--link",
        train_cfg["link"],
        "--seed",
        str(train_cfg["seed"]),
    ]
    cmd.extend(_as_flag(bool(train_cfg.get("no_lattice", False)), "--no_lattice"))
    cmd.extend(_as_flag(bool(train_cfg.get("use_angle", False)), "--use_angle"))
    cmd.extend(_as_flag(bool(train_cfg.get("weighted_loss", False)), "--weighted_loss"))
    cmd.extend(
        _as_flag(
            bool(train_cfg.get("per_bin_standardize", False)),
            "--per_bin_standardize",
        )
    )

    fermi_sigma = float(train_cfg.get("fermi_sigma", 0.0))
    if fermi_sigma > 0:
        cmd.extend(["--fermi_sigma", str(fermi_sigma)])

    fermi_bin_idx = int(train_cfg.get("fermi_bin_idx", -1))
    if fermi_bin_idx >= 0:
        cmd.extend(["--fermi_bin_idx", str(fermi_bin_idx)])

    for key in ("train_ratio", "val_ratio", "test_ratio"):
        ratio = float(train_cfg.get(key, 0.0))
        if ratio > 0:
            cmd.extend([f"--{key}", str(ratio)])

    run_cmd(cmd, mat_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified mainline entrypoint")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to TOML config")
    parser.add_argument(
        "--stage",
        choices=["pull_baselines", "preprocess", "train", "all"],
        default="all",
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--target",
        choices=["wly", "matformer", "both"],
        default="both",
        help="Which training target to run",
    )
    parser.add_argument(
        "--baseline",
        choices=["cgcnn", "alignn", "ours", "all"],
        default="all",
        help="Baseline source to pull when --stage pull_baselines",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    print(f"[CONFIG] {cfg_path}")

    if args.stage == "pull_baselines":
        run_pull_baselines(cfg, args.baseline)
        return

    if args.stage in ("preprocess", "all"):
        # Data preprocessing is only for WLY pipeline.
        run_wly_preprocess(cfg)

    if args.stage in ("train", "all"):
        if args.target in ("wly", "both"):
            run_wly_train(cfg)
        if args.target in ("matformer", "both"):
            run_matformer_train(cfg)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with exit code {e.returncode}", file=sys.stderr)
        raise
