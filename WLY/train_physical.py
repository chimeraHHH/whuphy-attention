import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset, Batch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Matformer-main')))

from WLY.physical_matformer import PhysicalMatformer, PhysicalMatformerConfig
from matformer.graphs import PygGraph
from jarvis.core.atoms import Atoms

class Imp2dDataset(Dataset):
    def __init__(self, data_path, transform=None):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        
        with open(data_path, 'r') as f:
            self.raw_data = json.load(f)
            
    def len(self):
        return len(self.raw_data)
    
    def get(self, idx):
        entry = self.raw_data[idx]
        
        # Reconstruct JARVIS Atoms object
        # Matformer uses jarvis.core.atoms to build graph
        atoms_dict = entry['atoms']
        atoms = Atoms(
            lattice_mat=atoms_dict['lattice'],
            elements=atoms_dict['elements'],
            coords=atoms_dict['positions'],
            cartesian=True
        )
        
        # Convert to PyG Data
        # Using PygGraph from matformer package
        # Note: PygGraph in this codebase doesn't take args in __init__, but in atom_dgl_multigraph
        # And PygGraph is a class, but we need to call the static method or instance method
        
        # Correct usage based on graphs.py analysis:
        # PygGraph.atom_dgl_multigraph is a static method
        data = PygGraph.atom_dgl_multigraph(
            atoms=atoms,
            neighbor_strategy='k-nearest',
            cutoff=8.0,
            max_neighbors=12,
            atom_features='cgcnn',
            use_canonize=True,
            compute_line_graph=False # We don't need line graph for now
        )
        
        # Add target
        target = torch.tensor(entry['target'], dtype=torch.float)
        data.y = target
        
        # Ensure lattice is present (Matformer PygGraph might not add it by default in all versions)
        # PhysicalMatformer expects 'lattice'
        if not hasattr(data, 'lattice'):
             data.lattice = torch.tensor(atoms.lattice_mat, dtype=torch.float).unsqueeze(0)
             
        # Ensure pos is present
        if not hasattr(data, 'pos'):
             data.pos = torch.tensor(atoms.cartesian_coords, dtype=torch.float)
             
        return data

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    mae_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        
        # Squeeze output to match target shape (batch_size)
        # output is (batch, 1), y is (batch)
        output = output.squeeze()
        
        # MSE Loss for regression
        loss = F.mse_loss(output, data.y)
        mae = F.l1_loss(output, data.y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        mae_loss += mae.item() * data.num_graphs
        
    return total_loss / len(loader.dataset), mae_loss / len(loader.dataset)

import matplotlib.pyplot as plt

def plot_history(history, save_path="WLY/training_history.png"):
    epochs = range(1, len(history['train_mse']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot MSE
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_mse'], 'b-', label='Train MSE')
    plt.plot(epochs, history['val_mse'], 'r-', label='Val MSE')
    plt.title('Mean Squared Error')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_mae'], 'b-', label='Train MAE')
    plt.plot(epochs, history['val_mae'], 'r-', label='Val MAE')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epochs')
    plt.ylabel('Error (eV/atom)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"History plot saved to {save_path}")
    plt.close()

def plot_parity(y_true, y_pred, save_path="WLY/parity_plot.png"):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    
    # Perfect fit line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    
    plt.xlabel('DFT Formation Energy (eV)')
    plt.ylabel('Predicted Formation Energy (eV)')
    plt.title(f'Parity Plot (MAE: {np.mean(np.abs(np.array(y_true) - np.array(y_pred))):.4f} eV)')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Parity plot saved to {save_path}")
    plt.close()

def evaluate(model, loader, device, return_preds=False):
    model.eval()
    total_loss = 0
    mae_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            output = output.squeeze()
            
            loss = F.mse_loss(output, data.y)
            mae = F.l1_loss(output, data.y)
            
            total_loss += loss.item() * data.num_graphs
            mae_loss += mae.item() * data.num_graphs
            
            if return_preds:
                all_preds.extend(output.cpu().numpy().tolist())
                all_targets.extend(data.y.cpu().numpy().tolist())
            
    avg_mse = total_loss / len(loader.dataset)
    avg_mae = mae_loss / len(loader.dataset)
    
    if return_preds:
        return avg_mse, avg_mae, all_preds, all_targets
    return avg_mse, avg_mae

def main():
    # Configuration
    DATA_FILE = os.path.join(os.path.dirname(__file__), 'imp2d_processed.json')
    BATCH_SIZE = 16 # Adjust based on GPU memory
    EPOCHS = 100
    LR = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file {DATA_FILE} not found. Please run process_imp2d.py first.")
        return

    print(f"Loading data from {DATA_FILE}...")
    dataset = Imp2dDataset(DATA_FILE)
    
    # Subset for quick demo: 250 training samples
    # We'll use a small validation set too, e.g., 50 samples
    subset_size = 300 
    if len(dataset) > subset_size:
        print(f"Subsetting dataset to {subset_size} samples for quick training...")
        dataset, _ = torch.utils.data.random_split(dataset, [subset_size, len(dataset) - subset_size])
    
    # Simple split
    train_size = 250
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Initialize Model
    # Determine PDOS dimension from first sample
    # sample_y = dataset[0].y
    # pdos_dim = sample_y.shape[0]
    # print(f"Detected PDOS dimension: {pdos_dim}")
    
    # For eform regression, output dimension is 1
    pdos_dim = 1
    
    config = PhysicalMatformerConfig(
        name="physical_matformer",
        use_lattice=True,
        use_angle=True,
        pdos_dim=pdos_dim,
        conv_layers=4, # Smaller model for testing
        node_features=128
    )
    
    model = PhysicalMatformer(config).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    history = {'train_mse': [], 'train_mae': [], 'val_mse': [], 'val_mae': []}
    
    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_mae = train(model, train_loader, optimizer, DEVICE)
        val_loss, val_mae = evaluate(model, val_loader, DEVICE)
        
        scheduler.step(val_loss)
        
        history['train_mse'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_mse'].append(val_loss)
        history['val_mae'].append(val_mae)
        
        print(f"Epoch {epoch:03d}: Train MSE: {train_loss:.4f}, MAE: {train_mae:.4f} | Val MSE: {val_loss:.4f}, MAE: {val_mae:.4f}")
        
    # Final Evaluation with predictions
    print("Generating plots...")
    plot_history(history)
    
    _, _, val_preds, val_targets = evaluate(model, val_loader, DEVICE, return_preds=True)
    plot_parity(val_targets, val_preds)
    print("Done.")

if __name__ == "__main__":
    main()
