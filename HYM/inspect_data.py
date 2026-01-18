
import torch
import os

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(WORK_DIR, "GNNs-PDOS", "shujujione", "JPCA2025", "dataset")
train_path = os.path.join(DATASET_DIR, "train_data.pth")

print("Loading...")
data = torch.load(train_path)
print(f"Loaded {len(data)} items.")
first = data[0]
print(f"Keys: {first.keys}")
print(f"mp_id: {first.mp_id if hasattr(first, 'mp_id') else 'Not Found'}")
print(f"structure_name: {first.structure_name if hasattr(first, 'structure_name') else 'Not Found'}")
