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

Run only Matformer training:

```bash
python mainline/run.py --stage train --target matformer
```
