# Baseline Ladder

This directory hosts baseline assets for:

1. `CGCNN`
2. `ALIGNN`
3. `Ours` (mainline path)

External upstream repos are pulled into:

- `mainline/baseline/third_party/cgcnn`
- `mainline/baseline/third_party/alignn`

Pull command:

```bash
python mainline/run.py --stage pull_baselines --baseline all
```

Notes:
- `third_party/` is ignored in root `.gitignore` to avoid accidental vendoring.
- `Ours` points to `mainline/` and is not cloned from an external repository.
