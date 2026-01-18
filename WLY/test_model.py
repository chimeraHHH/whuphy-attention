import torch
import sys
import os

# Add project root to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Matformer-main')))

from WLY.physical_matformer import PhysicalMatformer, PhysicalMatformerConfig
from torch_geometric.data import Data, Batch

def test_physical_matformer():
    print("Testing PhysicalMatformer initialization and forward pass...")
    
    # 1. Configuration
    config = PhysicalMatformerConfig(
        name="physical_matformer",
        use_lattice=True,
        use_angle=True,
        pdos_dim=200,
        node_features=128,
        edge_features=128
    )
    
    model = PhysicalMatformer(config)
    print("Model initialized successfully.")
    
    # 2. Create Dummy Data
    # Simulate a batch of 2 crystals
    
    # Crystal 1: 3 atoms
    pos1 = torch.rand(3, 3)
    # Simulate CGCNN features (92-dim float vectors) instead of just atom indices
    z1 = torch.randn(3, 92) 
    lattice1 = torch.eye(3).unsqueeze(0) # Simple cubic
    
    # Crystal 2: 2 atoms
    pos2 = torch.rand(2, 3)
    z2 = torch.randn(2, 92)
    lattice2 = torch.tensor([[5.0, 0, 0], [0, 5.0, 0], [0, 0, 10.0]]).unsqueeze(0)
    
    # Create Edges (Fully connected for simplicity)
    edge_index1 = torch.tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], dtype=torch.long)
    edge_attr1 = torch.rand(6, 3) # Random distances vectors
    
    edge_index2 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr2 = torch.rand(2, 3)
    
    data1 = Data(x=z1, edge_index=edge_index1, edge_attr=edge_attr1, pos=pos1, lattice=lattice1)
    data2 = Data(x=z2, edge_index=edge_index2, edge_attr=edge_attr2, pos=pos2, lattice=lattice2)
    
    # Batching
    batch_data = Batch.from_data_list([data1, data2])
    
    print(f"Input Batch: {batch_data}")
    
    # 3. Forward Pass
    try:
        output = model(batch_data)
        print("Forward pass successful.")
        print(f"Output Shape: {output.shape}")
        
        # 4. Verification
        expected_shape = (2, 200) # (batch_size, pdos_dim)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        print("✅ Output dimension check passed!")
        
        # Check for NaN
        if torch.isnan(output).any():
            print("❌ Output contains NaN values!")
        else:
            print("✅ No NaN values in output.")
            
    except Exception as e:
        print(f"❌ Forward pass failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_physical_matformer()
