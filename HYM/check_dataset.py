import torch
import os

dataset_path = 'GNNs-PDOS/shujujione/JPCA2025/dataset'
target_mp_ids = ['mp-1105800'] # LuB2C

found = []

for split in ['train_data.pth', 'val_data.pth', 'test_data.pth']:
    path = os.path.join(dataset_path, split)
    print(f"Checking {split}...")
    try:
        data_list = torch.load(path)
        for data in data_list:
            if data.mp_id in target_mp_ids:
                print(f"Found {data.mp_id} in {split}")
                found.append(data)
    except Exception as e:
        print(f"Error reading {split}: {e}")

if not found:
    print("Target materials not found in the downloaded dataset.")
else:
    print(f"Found {len(found)} target materials.")
