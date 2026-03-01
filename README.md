# whuphy-attention

## Mainline

Unified mainline lives in [`mainline/`](/e:/2026Winter/whuphy-attention/mainline):

- Data pipeline: `WLY` (`.pt -> .pkl -> graph -> filter`)
- Training entry: `WLY` and `Matformer-1` are both runnable from one command
- Single config file: `mainline/config.toml`

Run:

```bash
python mainline/run.py --stage all --target both
```

Useful options:

```bash
python mainline/run.py --stage split_lock --split_version v1
python mainline/run.py --stage pull_baselines --baseline all
python mainline/run.py --stage baseline --baseline train --baseline_model all --split_version v1
python mainline/run.py --stage baseline --baseline eval --baseline_model all --split_version v1
python mainline/run.py --stage preprocess --target wly
python mainline/run.py --stage train --target matformer
python mainline/run.py --config mainline/config.toml --stage train --target wly
```

## Branch Roles

- `mainline/`, `WLY/`, `Matformer-1/`: active mainline path.
- `BJQ/`, `HYM/`: experimental branches kept for comparison and archived experiments.
