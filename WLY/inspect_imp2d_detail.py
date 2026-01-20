
import sqlite3
import pickle
import json
import os
import sys

def inspect_db(db_path):
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = "SELECT unique_id, data, key_value_pairs FROM systems LIMIT 1"
    try:
        cursor.execute(query)
        row = cursor.fetchone()
        
        if row:
            uid, data_blob, kv_json = row
            print(f"--- Inspecting {uid} ---")
            
            # Inspect JSON metadata
            if kv_json:
                print("\n[Key Value Pairs (JSON)]:")
                try:
                    kv = json.loads(kv_json)
                    for k, v in kv.items():
                        print(f"  {k}: {v}")
                except:
                    print("  (Invalid JSON)")
            
            # Inspect Pickled Data
            if data_blob:
                print("\n[Pickled Data Keys]:")
                try:
                    data = pickle.loads(data_blob)
                    if isinstance(data, dict):
                        for k in data.keys():
                            val_type = type(data[k])
                            print(f"  {k}: {val_type}")
                            # Check for PDOS-like keys
                            if 'dos' in k.lower() or 'pdos' in k.lower():
                                print(f"    -> FOUND POTENTIAL PDOS: {k}")
                    else:
                        print(f"  Data is not a dict, it is: {type(data)}")
                except Exception as e:
                    print(f"  (Pickle Error: {e})")
            else:
                print("\n[Pickled Data]: None")
                
        else:
            print("Table 'systems' is empty.")
            
    except Exception as e:
        print(f"Database error: {e}")
        
    conn.close()

if __name__ == "__main__":
    # Check BJQ folder
    db_path = os.path.abspath("BJQ/imp2d.db")
    inspect_db(db_path)
