import os
import torch
import torch.nn as nn
import pandas as pd
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import json

# Add path to import model
sys.path.append(r'd:\Github hanjia\whuphy-attention\BJQ\GNNs-PDOS\models')
from Final_Project_Model import HierarchicalHybridMatformer
from data_inference import Imp2DDataset

# --- Configuration ---
DB_PATH = r'd:\Github hanjia\whuphy-attention\BJQ\imp2d.db'
ORB_PATH = r'd:\Github hanjia\whuphy-attention\BJQ\orbital_electrons.json'
MODEL_WEIGHTS_PATH = r'd:\Github hanjia\whuphy-attention\BJQ\GNNs-PDOS\shujujione\JPCA2025\CGT_phys_weights.pth'
OUTPUT_DIR = r'd:\Github hanjia\whuphy-attention\BJQ'
BATCH_SIZE = 32 # Adjust based on GPU memory
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_prediction():
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data
    print("Loading dataset from DB...")
    dataset_handler = Imp2DDataset(DB_PATH, ORB_PATH)
    # Load all data (or a subset for testing)
    data_list = dataset_handler.load_all(limit=None) 
    print(f"Total samples loaded: {len(data_list)}")
    
    loader = DataLoader(data_list, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Initialize Model
    model = HierarchicalHybridMatformer(
        node_input_dim=118,
        edge_dim=128,
        hidden_dim=256,
        out_dim=201*4,
        num_layers=3,
        heads=4,
        dropout=0.1,
        use_vacancy_feature=True # Enable our new feature
    ).to(DEVICE)
    
    # Try loading weights
    try:
        state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE)
        # Handle potential key mismatches if weights are from DataParallel or different version
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '') # remove 'module.'
            new_state_dict[name] = v
            
        # Load with strict=False to ignore missing vacancy weights in the checkpoint
        model.load_state_dict(new_state_dict, strict=False)
        print("Weights loaded successfully (partial load if architecture changed).")
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Proceeding with random initialization (untrained model) - WARNING: Predictions will be random!")

    model.eval()
    
    results = []
    pdos_results = {} # Store PDOS for plotting (key: mp_id)
    
    print("Starting inference...")
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(DEVICE)
            
            # Forward pass
            out_pdos, out_scalar = model(batch)
            
            # Process results
            # out_scalar: [Batch, 3] -> Formation Energy, Band Gap, p-band Center
            # out_pdos: [Batch, 4, 201]
            
            batch_size_curr = batch.num_graphs
            for i in range(batch_size_curr):
                mp_id = batch.mp_id[i]
                
                # Scalar predictions
                pred_formation_energy = out_scalar[i, 0].item()
                pred_band_gap = out_scalar[i, 1].item()
                pred_p_band_center = out_scalar[i, 2].item()
                
                # Ground Truth (if available)
                # Note: formation_energy in batch is [Batch, 1]
                true_formation_energy = batch.formation_energy[i].item()
                true_band_gap = batch.band_gap[i].item()
                
                results.append({
                    'mp_id': mp_id,
                    'pred_formation_energy': pred_formation_energy,
                    'pred_band_gap': pred_band_gap,
                    'pred_p_band_center': pred_p_band_center,
                    'true_formation_energy': true_formation_energy,
                    'true_band_gap': true_band_gap
                })
                
                # Store PDOS for a few samples (e.g., first 6 for Figure 4)
                if len(pdos_results) < 6:
                    # out_pdos: [Batch, 4, 201] -> [4, 201]
                    pdos_data = out_pdos[i].cpu().numpy().tolist()
                    pdos_results[mp_id] = pdos_data

    # Save Scalar Results
    df = pd.DataFrame(results)
    output_file = os.path.join(OUTPUT_DIR, 'prediction_results.csv')
    df.to_csv(output_file, index=False)
    print(f"Scalar predictions saved to {output_file}")
    
    # Save PDOS Results (JSON)
    pdos_file = os.path.join(OUTPUT_DIR, 'prediction_pdos_sample.json')
    with open(pdos_file, 'w') as f:
        json.dump(pdos_results, f)
    print(f"Sample PDOS predictions saved to {pdos_file}")

if __name__ == '__main__':
    run_prediction()
