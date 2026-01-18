
import torch
import os
import numpy as np

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(WORK_DIR, "GNNs-PDOS", "shujujione", "JPCA2025", "dataset")
train_path = os.path.join(DATASET_DIR, "train_data.pth")

print(f"Loading {train_path}...")
try:
    data = torch.load(train_path)
    print(f"Loaded. Type: {type(data)}")
    print(f"Length: {len(data)}")
    
    indices = np.random.choice(len(data), 10, replace=False)
    print(f"Sample indices: {indices}")
    subset = [data[i] for i in indices]
    print("Subset created.")
except Exception as e:
    print(f"Error: {e}")
