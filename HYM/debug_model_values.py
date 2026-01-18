
import os
import torch
import numpy as np
from defect_prediction_model import CGT
import json

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(WORK_DIR, "GNNs-PDOS", "shujujione", "JPCA2025", "CGT_phys_weights.pth")
MP_PDOS_DATA = os.path.join(WORK_DIR, "mp_pdos_data.json")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Load Model
    model = CGT(edge_dim=14, out_dim=201*4).to(device)
    
    # 2. Check Weights Loading
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Weights not found at {WEIGHTS_PATH}")
        return

    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # Remove module. prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    # Load and check missing keys
    keys = model.load_state_dict(new_state_dict, strict=False)
    print(f"Missing keys: {keys.missing_keys}")
    print(f"Unexpected keys: {keys.unexpected_keys}")

    # 3. Dummy Input Test (Simulate LuB2C)
    # 4 atoms (e.g. Lu, B, B, C)
    # 14 edge features
    # 4 orbital counts
    
    # Create random input
    num_nodes = 4
    x = torch.tensor([71, 5, 5, 6], dtype=torch.long).to(device) # Lu=71, B=5, C=6
    # Note: If embedding is 118, max index is 117. 71 is safe.
    
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long).to(device)
    edge_attr = torch.randn(8, 14).to(device)
    energies = torch.linspace(-10, 10, 201).unsqueeze(0).to(device)
    orbital_counts = torch.randn(4, 4).to(device) # [num_nodes, 4]
    
    from torch_geometric.data import Data, Batch
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, energies=energies, orbital_counts=orbital_counts)
    batch = Batch.from_data_list([data]).to(device)
    
    model.eval()
    with torch.no_grad():
        out = model(batch) # [1, 4, 201]
        out_np = out.cpu().numpy()
        
    print("\nPrediction Statistics:")
    print(f"Shape: {out_np.shape}")
    print(f"Min: {out_np.min()}")
    print(f"Max: {out_np.max()}")
    print(f"Mean: {out_np.mean()}")
    print(f"Std: {out_np.std()}")
    
    # Compare with Real Data Magnitude if available
    if os.path.exists(MP_PDOS_DATA):
        with open(MP_PDOS_DATA, 'r') as f:
            mp_data = json.load(f)
            first_key = list(mp_data.keys())[0]
            real_dos = mp_data[first_key]['orbitals']['s'] # Just take 's'
            # Note: this is total DOS. We need per-atom.
            # Assuming ~4 atoms
            real_per_atom = np.array(real_dos) / 4.0
            print("\nReal Data (Approx Per-Atom) Statistics:")
            print(f"Max: {real_per_atom.max()}")
            print(f"Mean: {real_per_atom.mean()}")

if __name__ == "__main__":
    main()
