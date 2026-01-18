import sqlite3
import json
import struct
import pickle
import numpy as np
import os
from ase import Atoms
from ase.io import write

def decode_blob(blob, dtype='d'):
    """Decode binary blob to list of numbers."""
    if not blob: return []
    try:
        # Calculate number of elements
        element_size = struct.calcsize(dtype)
        count = len(blob) // element_size
        return struct.unpack(f"{count}{dtype}", blob)
    except Exception as e:
        print(f"Decode error: {e}")
        return []

def process_imp2d_db(db_path, output_path):
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query essential fields
    # Note: 'numbers' is atomic numbers, 'positions' is coordinates, 'cell' is lattice
    # 'data' column typically holds the extra properties like PDOS in a pickled dict
    query = """
    SELECT unique_id, numbers, positions, cell, data, key_value_pairs 
    FROM systems
    """
    
    try:
        cursor.execute(query)
    except sqlite3.OperationalError as e:
        print(f"Query failed: {e}")
        print("Checking table info...")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables found: {tables}")
        return

    processed_data = []
    skipped_count = 0
    
    print("Iterating through database entries...")
    for row in cursor:
        uid, numbers_blob, pos_blob, cell_blob, data_blob, kv_json = row
        
        try:
            # 1. Decode Structure
            # Atomic numbers are typically integers ('i' or 'q'), ASE db uses 'i' often but let's check
            # Try 'i' first
            try:
                 numbers = decode_blob(numbers_blob, 'i')
            except:
                 numbers = decode_blob(numbers_blob, 'q')
            
            # If numbers are empty, something is wrong
            if not numbers:
                # print(f"Empty numbers for {uid}")
                pass

            # Positions (flattened array -> N x 3)
            pos_flat = decode_blob(pos_blob, 'd')
            num_atoms = len(numbers)
            # if len(pos_flat) != num_atoms * 3:
                 # print(f"Position mismatch for {uid}: {len(pos_flat)} vs {num_atoms*3}")

            positions = [pos_flat[i*3:(i+1)*3] for i in range(num_atoms)]
            
            # Lattice (flattened array -> 3 x 3)
            cell_flat = decode_blob(cell_blob, 'd')
            cell = [cell_flat[i*3:(i+1)*3] for i in range(3)]
            
            # 2. Extract Target (eform)
            target_val = None
            
            # Priority 1: Check key_value_pairs (JSON) - eform is usually here
            if kv_json:
                try:
                    kv_data = json.loads(kv_json)
                    if 'eform' in kv_data:
                        target_val = kv_data['eform']
                except Exception as e:
                    print(f"JSON load error for {uid}: {e}")
            
            # Priority 2: Check data (Pickle)
            if target_val is None and data_blob:
                try:
                    extra_data = pickle.loads(data_blob)
                    if 'eform' in extra_data:
                        target_val = extra_data['eform']
                except:
                    pass
            
            if target_val is not None:
                # Ensure it's a float
                target_val = float(target_val)
                
                # Filter out NaN values
                if np.isnan(target_val):
                    # print(f"Skipping {uid}: eform is NaN")
                    continue
                
                entry = {
                    "id": uid,
                    "atoms": {
                        "elements": numbers,
                        "positions": positions,
                        "lattice": cell
                    },
                    "target": target_val
                }
                processed_data.append(entry)
            else:
                skipped_count += 1
                if skipped_count < 5: 
                     print(f"Skipping {uid}: No 'eform' found.")

        except Exception as e:
            print(f"Error processing row {uid}: {e}")
            continue

    conn.close()
    
    print(f"Processing complete.")
    print(f"Total entries extracted: {len(processed_data)}")
    print(f"Skipped entries: {skipped_count}")
    
    if len(processed_data) > 0:
        # Check target stats
        sample_target = processed_data[0]['target']
        print(f"Sample target (eform): {sample_target}")

        print(f"Saving to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(processed_data, f)
        print("Done.")
    else:
        print("No valid data found. Please check database content.")

if __name__ == "__main__":
    db_path = os.path.join(os.path.dirname(__file__), 'imp2d.db')
    output_path = os.path.join(os.path.dirname(__file__), 'imp2d_processed.json')
    process_imp2d_db(db_path, output_path)
