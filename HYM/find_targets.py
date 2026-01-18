
import torch
import os

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(WORK_DIR, "GNNs-PDOS", "shujujione", "JPCA2025", "dataset")
train_path = os.path.join(DATASET_DIR, "train_data.pth")
test_path = os.path.join(DATASET_DIR, "test_data.pth")

target_ids = ["mp-1189682", "mp-1220094", "mp-1215264", "mp-1187277", "mp-1184402", "mp-754261"]

def check_file(path, name):
    if not os.path.exists(path): return
    print(f"Checking {name}...")
    try:
        data = torch.load(path)
        found = 0
        for d in data:
            if hasattr(d, 'mp_id') and d.mp_id in target_ids:
                print(f"Found {d.mp_id} in {name}")
                found += 1
        print(f"Total found in {name}: {found}")
    except Exception as e:
        print(f"Error: {e}")

check_file(train_path, "train_data")
check_file(test_path, "test_data")
