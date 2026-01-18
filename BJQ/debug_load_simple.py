
print("Start debug script", flush=True)
import torch
print("Imported torch", flush=True)
import os
print("Imported os", flush=True)

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(WORK_DIR, "GNNs-PDOS", "shujujione", "JPCA2025", "dataset")
files = ["train_data.pth"]

for fname in files:
    fpath = os.path.join(DATASET_DIR, fname)
    if os.path.exists(fpath):
        print(f"Loading {fname}...", flush=True)
        try:
            ds = torch.load(fpath)
            print(f"Loaded {len(ds)} items", flush=True)
        except Exception as e:
            print(f"Error: {e}", flush=True)
print("Done", flush=True)
