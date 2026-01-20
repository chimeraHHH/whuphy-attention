
import torch
import glob
import os

def check_data():
    files = glob.glob("WLY/dataset/*.pt")
    if not files:
        print("No data files found.")
        return
    
    data = torch.load(files[0])
    print("Keys in data:", data.keys)
    if hasattr(data, 'pos'):
        print("Pos shape:", data.pos.shape)
    if hasattr(data, 'lattice'):
        print("Lattice shape:", data.lattice.shape)
        print("Lattice:", data.lattice)
    if hasattr(data, 'x'):
        print("X shape:", data.x.shape)
    if hasattr(data, 'defect_mask'):
        print("Defect mask found.")
    else:
        print("Defect mask NOT found.")

if __name__ == "__main__":
    check_data()
