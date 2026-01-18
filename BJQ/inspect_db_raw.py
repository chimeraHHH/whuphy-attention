
import sqlite3
import json

db_path = "imp2d.db"

print(f"Connecting to {db_path} with raw sqlite...")
try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Check 'keys' table to see what properties are stored
    print("\n--- Available Keys (Properties) ---")
    cursor.execute("SELECT * FROM keys")
    keys = cursor.fetchall()
    key_names = [k[0] for k in keys]
    print(key_names)
    
    # 2. Sample some data from key_value_pairs in systems table if it exists
    # Or check number_key_values / text_key_values tables which ASE uses
    
    print("\n--- Checking number_key_values table ---")
    cursor.execute("SELECT key, value FROM number_key_values LIMIT 20")
    num_vals = cursor.fetchall()
    for k, v in num_vals:
        print(f"{k}: {v}")
        
    print("\n--- Checking text_key_values table ---")
    cursor.execute("SELECT key, value FROM text_key_values LIMIT 20")
    text_vals = cursor.fetchall()
    for k, v in text_vals:
        print(f"{k}: {v}")

except Exception as e:
    print(f"Error: {e}")
