# Mainline Runner

This folder is the single entrypoint for the project mainline:

- `WLY` data pipeline
- `WLY` training/inference loop
- `Matformer-1` EDOS/PDOS training script

## Files

- `config.toml`: unified config
- `run.py`: unified CLI entry

## Quick Start

Run full pipeline and both trainers:

```bash
python mainline/run.py --stage all --target both
```

Note: `matformer.train.enabled` is `false` by default in `config.toml`.
Set it to `true` before running `--target matformer` or `--target both`.

Run only data preprocessing:

```bash
python mainline/run.py --stage preprocess --target wly
```

Pull baseline ladder sources (CGCNN / ALIGNN / Ours):

```bash
python mainline/run.py --stage pull_baselines --baseline all
```

Generate locked split artifacts (`80/10/10` with fixed seed):

```bash
python mainline/run.py --stage split_lock --split_version v1
```

Unified baseline orchestration:

```bash
python mainline/run.py --stage baseline --baseline train --baseline_model all --split_version v1
python mainline/run.py --stage baseline --baseline eval --baseline_model all --split_version v1
```

Run only Matformer training:

```bash
python mainline/run.py --stage train --target matformer
```
