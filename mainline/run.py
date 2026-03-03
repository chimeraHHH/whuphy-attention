#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.toml"
BASELINE_MODELS = ("cgcnn", "alignn", "ours")
BASELINE_ACTIONS = ("pull", "train", "eval")


def load_config(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print(f"[RUN] cwd={cwd} :: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _as_flag(value: bool, flag: str) -> list[str]:
    return [flag] if value else []


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _split_version(cfg: dict, override: str | None) -> str:
    if override:
        return override
    return str(cfg.get("baseline", {}).get("split_version", "v1"))


def _split_dir(cfg: dict, split_version: str) -> Path:
    split_cfg = cfg["baseline"]["split"]
    return _resolve_path(split_cfg["output_root"]) / split_version


def _split_ready(split_dir: Path) -> bool:
    required = ("train_indices.json", "val_indices.json", "test_indices.json")
    return all((split_dir / name).exists() for name in required)


def ensure_split_lock(cfg: dict, split_version: str) -> Path:
    split_cfg = cfg["baseline"]["split"]
    split_dir = _split_dir(cfg, split_version)
    if _split_ready(split_dir):
        return split_dir

    if not bool(split_cfg.get("auto_generate", True)):
        raise FileNotFoundError(
            f"Split artifacts not found at {split_dir} and auto_generate=false."
        )

    print(f"[SPLIT] Missing split artifacts at {split_dir}, generating now...")
    run_split_lock(cfg, split_version)
    if not _split_ready(split_dir):
        raise RuntimeError(f"Split generation failed, artifacts still missing: {split_dir}")
    return split_dir


def _selected_models(selector: str) -> list[str]:
    sel = selector.lower()
    if sel == "all":
        return list(BASELINE_MODELS)
    if sel not in BASELINE_MODELS:
        raise ValueError(
            f"Unknown baseline model '{selector}'. Use one of: all/cgcnn/alignn/ours."
        )
    return [sel]


def _pick_sample_id(sample: dict, idx: int, keys: list[str]) -> tuple[str, str]:
    for key in keys:
        val = sample.get(key)
        if val is not None and str(val) != "":
            return str(val), key
    return str(idx), "__index__"


def run_split_lock(cfg: dict, split_version: str) -> None:
    split_cfg = cfg["baseline"]["split"]
    data_path = _resolve_path(split_cfg["data"])
    if not data_path.exists():
        raise FileNotFoundError(f"Split-lock data file not found: {data_path}")

    train_ratio = float(split_cfg.get("train_ratio", 0.8))
    val_ratio = float(split_cfg.get("val_ratio", 0.1))
    test_ratio = float(split_cfg.get("test_ratio", 0.1))
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(
            f"train/val/test ratio must sum to 1.0, got {ratio_sum:.8f}."
        )

    seed = int(split_cfg.get("seed", 42))
    id_keys = list(split_cfg.get("id_keys", ["id", "unique_id", "uid", "mp_id"]))
    out_dir = _split_dir(cfg, split_version)
    manifest_root = _resolve_path(split_cfg.get("manifest_root", "mainline/baseline/manifests"))
    manifest_path = manifest_root / f"data_manifest_{split_version}.json"

    with data_path.open("rb") as f:
        dataset = pickle.load(f)
    if not isinstance(dataset, list):
        raise TypeError(f"Expected list dataset in {data_path}, got {type(dataset)}")
    n = len(dataset)
    if n == 0:
        raise ValueError(f"Dataset is empty: {data_path}")

    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    if len(test_indices) != n_test:
        raise RuntimeError("Split size mismatch after slicing.")
    if len(set(train_indices) & set(val_indices)) > 0:
        raise RuntimeError("train/val split overlap detected.")
    if len(set(train_indices) & set(test_indices)) > 0:
        raise RuntimeError("train/test split overlap detected.")
    if len(set(val_indices) & set(test_indices)) > 0:
        raise RuntimeError("val/test split overlap detected.")

    def ids_for(idxs: list[int]) -> list[str]:
        return [_pick_sample_id(dataset[i], i, id_keys)[0] for i in idxs]

    train_ids = ids_for(train_indices)
    val_ids = ids_for(val_indices)
    test_ids = ids_for(test_indices)

    out_dir.mkdir(parents=True, exist_ok=True)
    _save_json(out_dir / "train_indices.json", train_indices)
    _save_json(out_dir / "val_indices.json", val_indices)
    _save_json(out_dir / "test_indices.json", test_indices)
    _save_json(out_dir / "train_ids.json", train_ids)
    _save_json(out_dir / "val_ids.json", val_ids)
    _save_json(out_dir / "test_ids.json", test_ids)
    _save_json(
        out_dir / "split_meta.json",
        {
            "version": split_version,
            "seed": seed,
            "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
            "counts": {"total": n, "train": n_train, "val": n_val, "test": n_test},
            "id_keys": id_keys,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_file": str(data_path),
        },
    )

    write_toy = bool(split_cfg.get("write_toy", True))
    toy_size = int(split_cfg.get("toy_size", 200))
    if write_toy and toy_size > 0:
        toy_indices = train_indices[: min(toy_size, len(train_indices))]
        _save_json(out_dir / "toy_indices.json", toy_indices)
        _save_json(out_dir / "toy_ids.json", ids_for(toy_indices))

    manifest_root.mkdir(parents=True, exist_ok=True)
    _save_json(
        manifest_path,
        {
            "split_version": split_version,
            "data_file": str(data_path),
            "data_sha256": _sha256(data_path),
            "counts": {"total": n, "train": n_train, "val": n_val, "test": n_test},
            "seed": seed,
            "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
            "split_dir": str(out_dir),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    print(f"[SPLIT] version={split_version} total={n} train={n_train} val={n_val} test={n_test}")
    print(f"[SPLIT] artifacts: {out_dir}")
    print(f"[SPLIT] manifest:  {manifest_path}")


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


def run_pull_baselines(cfg: dict, model_selector: str) -> None:
    pull_cfg = cfg["baseline"]["pull"]
    root_dir = _resolve_path(pull_cfg["root_dir"])
    depth = int(pull_cfg.get("depth", 1))

    for name in _selected_models(model_selector):
        bcfg = pull_cfg[name]
        if not bool(bcfg.get("enabled", True)):
            print(f"[SKIP] baseline '{name}' disabled in config.")
            continue

        if name == "ours":
            ours_path = _resolve_path(bcfg.get("path", "mainline"))
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
    wly_dir = _resolve_path(cfg["paths"]["wly_dir"])
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


def run_wly_train(
    cfg: dict,
    *,
    split_dir: Path | None = None,
    output_dir_override: str | None = None,
) -> None:
    train_cfg = cfg["wly"]["train"]
    if not bool(train_cfg.get("enabled", True)):
        print("[SKIP] WLY training disabled in config.")
        return

    py = cfg["paths"]["python"]
    wly_dir = _resolve_path(cfg["paths"]["wly_dir"])
    output_dir = output_dir_override or train_cfg["output_dir"]

    cmd = [
        py,
        "train.py",
        "--data",
        train_cfg["data"],
        "--features",
        train_cfg["features"],
        "--output_dir",
        output_dir,
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
    if split_dir is not None:
        cmd.extend(["--split_dir", str(split_dir)])
    cmd.extend(_as_flag(bool(train_cfg.get("pin_memory", False)), "--pin_memory"))
    cmd.extend(_as_flag(bool(train_cfg.get("fp16", False)), "--fp16"))
    cmd.extend(_as_flag(bool(train_cfg.get("distributed", False)), "--distributed"))
    run_cmd(cmd, wly_dir)


def run_ours_baseline_train(cfg: dict, split_version: str) -> None:
    ours_cfg = cfg["baseline"]["models"]["ours"]
    if not bool(ours_cfg.get("enabled", True)):
        print("[SKIP] Ours baseline disabled in config.")
        return

    version = str(ours_cfg.get("split_version", split_version))
    split_dir = ensure_split_lock(cfg, version)
    output_dir = _resolve_path(ours_cfg.get("train_output_dir", "WLY/checkpoints_baseline"))
    run_wly_train(cfg, split_dir=split_dir, output_dir_override=str(output_dir))


def run_ours_baseline_eval(cfg: dict, split_version: str) -> None:
    ours_cfg = cfg["baseline"]["models"]["ours"]
    if not bool(ours_cfg.get("enabled", True)):
        print("[SKIP] Ours baseline disabled in config.")
        return

    py = cfg["paths"]["python"]
    wly_dir = _resolve_path(cfg["paths"]["wly_dir"])
    wly_train = cfg["wly"]["train"]
    version = str(ours_cfg.get("split_version", split_version))
    split_dir = ensure_split_lock(cfg, version)

    default_train_out = _resolve_path(ours_cfg.get("train_output_dir", "WLY/checkpoints_baseline"))
    ckpt_name = str(ours_cfg.get("eval_ckpt", "best_model.pth"))
    ckpt_path = _resolve_path(
        str(ours_cfg.get("eval_ckpt_path", str(default_train_out / ckpt_name)))
    )
    plot_path = _resolve_path(
        str(ours_cfg.get("plot_path", f"mainline/baseline/reports/parity_ours_{version}.png"))
    )
    metrics_out = _resolve_path(
        str(ours_cfg.get("metrics_out", f"mainline/baseline/reports/metrics_ours_{version}.json"))
    )

    cmd = [
        py,
        "tets_all.py",
        "--ckpt",
        str(ckpt_path),
        "--data",
        wly_train["data"],
        "--features",
        wly_train["features"],
        "--batch_size",
        str(wly_train["batch_size"]),
        "--num_workers",
        str(wly_train["num_workers"]),
        "--split_dir",
        str(split_dir),
        "--plot_path",
        str(plot_path),
        "--metrics_out",
        str(metrics_out),
    ]
    cmd.extend(_as_flag(bool(wly_train.get("pin_memory", False)), "--pin_memory"))
    run_cmd(cmd, wly_dir)


def run_external_baseline(
    cfg: dict,
    *,
    name: str,
    action: str,
    split_version: str,
) -> None:
    model_cfg = cfg["baseline"]["models"][name]
    if not bool(model_cfg.get("enabled", True)):
        print(f"[SKIP] baseline '{name}' disabled in config.")
        return

    cmd_key = f"{action}_cmd"
    cmd = list(model_cfg.get(cmd_key, []))
    if not cmd:
        print(
            f"[SKIP] baseline '{name}' has empty '{cmd_key}' in config; "
            "please fill it before running."
        )
        return

    split_dir = ensure_split_lock(cfg, split_version)
    replacements = {
        "root": str(ROOT),
        "split_dir": str(split_dir),
        "split_version": split_version,
    }
    formatted_cmd = [str(part).format(**replacements) for part in cmd]
    workdir = _resolve_path(str(model_cfg.get("workdir", ".")))
    run_cmd(formatted_cmd, workdir)


def run_baseline(cfg: dict, *, action: str, model_selector: str, split_version: str) -> None:
    if action not in BASELINE_ACTIONS:
        raise ValueError(
            f"Unknown baseline action '{action}'. "
            f"Use one of: {', '.join(BASELINE_ACTIONS)}."
        )

    models = _selected_models(model_selector)
    for name in models:
        print(f"[BASELINE] action={action} model={name}")
        if action == "pull":
            run_pull_baselines(cfg, name)
            continue
        if name == "ours":
            if action == "train":
                run_ours_baseline_train(cfg, split_version)
            elif action == "eval":
                run_ours_baseline_eval(cfg, split_version)
            continue
        run_external_baseline(cfg, name=name, action=action, split_version=split_version)


def run_matformer_train(cfg: dict) -> None:
    train_cfg = cfg["matformer"]["train"]
    if not bool(train_cfg.get("enabled", False)):
        print("[SKIP] Matformer training disabled in config.")
        return

    py = cfg["paths"]["python"]
    mat_dir = _resolve_path(cfg["paths"]["matformer_dir"])

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
        choices=[
            "split_lock",
            "pull_baselines",
            "baseline",
            "preprocess",
            "train",
            "all",
        ],
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
        default="all",
        help=(
            "For --stage baseline: action (pull/train/eval). "
            "For --stage pull_baselines: model selector (cgcnn/alignn/ours/all)."
        ),
    )
    parser.add_argument(
        "--baseline_model",
        choices=["cgcnn", "alignn", "ours", "all"],
        default="all",
        help="Model selector for --stage baseline",
    )
    parser.add_argument(
        "--split_version",
        default="",
        help="Split-lock version, e.g. v1. Defaults to baseline.split_version in config.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    print(f"[CONFIG] {cfg_path}")

    split_version = _split_version(cfg, args.split_version or None)

    if args.stage == "split_lock":
        run_split_lock(cfg, split_version)
        return

    if args.stage == "pull_baselines":
        model_selector = args.baseline.lower()
        run_pull_baselines(cfg, model_selector)
        return

    if args.stage == "baseline":
        action = args.baseline.lower()
        run_baseline(
            cfg,
            action=action,
            model_selector=args.baseline_model.lower(),
            split_version=split_version,
        )
        return

    if args.stage in ("preprocess", "all"):
        run_wly_preprocess(cfg)

    if args.stage in ("train", "all"):
        if args.target in ("wly", "both"):
            run_wly_train(cfg)
        if args.target in ("matformer", "both"):
            run_matformer_train(cfg)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, TypeError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with exit code {e.returncode}", file=sys.stderr)
        raise
