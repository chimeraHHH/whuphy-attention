import torch
import os
try:
    path = r"d:\Github hanjia\whuphy-attention\BJQ\GNNs-PDOS\shujujione\JPCA2025\dataset\val_data.pth"
    data = torch.load(path)
    print("Loaded val data")
    print("Keys:", data[0].keys)
    if hasattr(data[0], 'formation_energy'):
        print("formation_energy found")
    else:
        print("formation_energy NOT found")
except Exception as e:
    print(e)
