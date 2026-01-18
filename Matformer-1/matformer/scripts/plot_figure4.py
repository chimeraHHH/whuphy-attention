import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_predictions(run_dir):
    fp = os.path.join(run_dir, "multi_out_predictions.json")
    if not os.path.exists(fp):
        return []
    with open(fp, "r") as f:
        data = json.load(f)
    return data

def sample_indices(entries, k=2):
    maes = []
    for i, e in enumerate(entries):
        y = np.array(e.get("target", []), dtype=float)
        p = np.array(e.get("predictions", []), dtype=float)
        if y.size == 0 or p.size == 0:
            maes.append((i, np.inf))
        else:
            m = float(np.mean(np.abs(y - p)))
            maes.append((i, m))
    if not maes:
        return []
    maes.sort(key=lambda x: x[1])
    idxs = [maes[0][0]]
    if len(maes) > 1:
        idxs.append(maes[-1][0])
    return idxs[:k]

def plot_entry(ax, entry, title):
    y = np.array(entry.get("target", []), dtype=float)
    p = np.array(entry.get("predictions", []), dtype=float)
    x = np.arange(len(y))
    ax.plot(x, y, label="target", color="#1f77b4")
    ax.plot(x, p, label="prediction", color="#ff7f0e")
    ax.set_title(title)
    ax.set_xlabel("Energy bin")
    ax.set_ylabel("DOS")
    ax.grid(True, alpha=0.3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edos_dir", default=os.path.join(".", "runs_quick", "edos_up_log"))
    parser.add_argument("--pdos_dir", default=os.path.join(".", "runs_quick", "pdos_elast_log"))
    parser.add_argument("--output", default=os.path.join(".", "runs_quick", "figure4.png"))
    args = parser.parse_args()

    edos_entries = load_predictions(args.edos_dir)
    pdos_entries = load_predictions(args.pdos_dir)

    edos_idxs = sample_indices(edos_entries, k=2)
    pdos_idxs = sample_indices(pdos_entries, k=2)

    n_rows = 2
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6))

    for j, idx in enumerate(edos_idxs):
        entry = edos_entries[idx]
        title = f"EDOS id={entry.get('id','')}"
        plot_entry(axes[0, j], entry, title)
        axes[0, j].legend()

    for j, idx in enumerate(pdos_idxs):
        entry = pdos_entries[idx]
        title = f"pDOS id={entry.get('id','')}"
        plot_entry(axes[1, j], entry, title)
        axes[1, j].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=200)
    plt.close()
    print("saved:", args.output)

if __name__ == "__main__":
    main()