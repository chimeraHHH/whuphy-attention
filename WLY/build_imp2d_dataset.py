
import sqlite3
import json
import torch
import struct
import os
import numpy as np
from torch_geometric.data import Data
from pymatgen.core.periodic_table import Element

# Ensure output directory exists
OUTPUT_DIR = "WLY/dataset_imp2d"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def decode_blob(blob, dtype='d'):
    """Decode binary blob to list of numbers."""
    if not blob: return []
    try:
        element_size = struct.calcsize(dtype)
        count = len(blob) // element_size
        return struct.unpack(f"{count}{dtype}", blob)
    except:
        return []

def get_defect_mask(numbers, dopant_symbol):
    """
    Identify defect atoms based on dopant symbol.
    Returns a tensor of 0s and 1s.
    """
    mask = torch.zeros(len(numbers), dtype=torch.long)
    
    if not dopant_symbol or dopant_symbol.lower() == 'none':
        return mask
        
    try:
        dopant_z = Element(dopant_symbol).Z
        # Find indices where atomic number matches dopant
        # Note: This simple heuristic assumes the host doesn't contain the dopant element.
        # For complex cases, we'd need more metadata.
        for i, z in enumerate(numbers):
            if z == dopant_z:
                mask[i] = 1
    except:
        pass
        
    return mask

def process_db(db_path):
    print(f"Processing {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = "SELECT unique_id, numbers, positions, cell, key_value_pairs FROM systems"
    cursor.execute(query)
    
    count = 0
    for row in cursor:
        uid, numbers_blob, pos_blob, cell_blob, kv_json = row
        
        try:
            # Decode Structure
            try:
                numbers = decode_blob(numbers_blob, 'i')
            except:
                numbers = decode_blob(numbers_blob, 'q')
                
            if not numbers: continue
            
            num_atoms = len(numbers)
            pos_flat = decode_blob(pos_blob, 'd')
            positions = torch.tensor(pos_flat, dtype=torch.float).view(num_atoms, 3)
            
            cell_flat = decode_blob(cell_blob, 'd')
            lattice = torch.tensor(cell_flat, dtype=torch.float).view(3, 3).unsqueeze(0) # (1, 3, 3)
            
            # Metadata
            dopant = None
            target_eform = None
            
            if kv_json:
                kv = json.loads(kv_json)
                dopant = kv.get('dopant')
                target_eform = kv.get('eform')
            
            # Filter valid targets
            if target_eform is None or np.isnan(target_eform):
                continue
                
            # Create Defect Mask
            defect_mask = get_defect_mask(numbers, dopant)
            
            # Node Features
            x = torch.tensor(numbers, dtype=torch.long).unsqueeze(1)
            
            # Edge Features (Simple k-NN or Radius)
            # We can compute this on-the-fly or precompute.
            # Here we just save positions, let the model/loader compute edges if needed,
            # OR we compute a simple radius graph here.
            # For compatibility with DAGL, let's compute edges.
            from torch_geometric.nn import radius_graph
            edge_index = radius_graph(positions, r=8.0, loop=False)
            
            # Edge Attr (Distance)
            src, dst = edge_index
            dist = (positions[src] - positions[dst]).norm(dim=-1).unsqueeze(1)
            
            # RBF (reuse definition or simplified)
            # We skip complex RBF here to keep script standalone; 
            # DAGL model can re-compute or we assume simple RBF.
            # Let's save raw distance as edge_attr for now.
            edge_attr = dist
            
            # Energies/Orbital Counts (Missing in Imp2d?)
            # We fill with zeros or dummy for now as placeholders
            energies = torch.zeros(1, 201) 
            orbital_counts = torch.zeros(num_atoms, 4)
            
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=positions,
                lattice=lattice,
                defect_mask=defect_mask,
                y=torch.tensor([target_eform], dtype=torch.float).view(1, 1), # Target
                energies=energies,
                orbital_counts=orbital_counts,
                mp_id=uid
            )
            
            torch.save(data, os.path.join(OUTPUT_DIR, f"{uid}.pt"))
            count += 1
            
            if count % 100 == 0:
                print(f"Processed {count} samples...")
                
        except Exception as e:
            print(f"Error {uid}: {e}")
            
    print(f"Done. Saved {count} files to {OUTPUT_DIR}")

if __name__ == "__main__":
    db_path = os.path.abspath("BJQ/imp2d.db")
    process_db(db_path)
