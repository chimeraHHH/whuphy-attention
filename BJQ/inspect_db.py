
import sqlite3
from ase.db import connect
import json

db_path = "imp2d.db"

print(f"Connecting to {db_path}...")
try:
    # Try using ASE to connect as it is likely an ASE database
    db = connect(db_path)
    print(f"Database loaded. Total rows: {len(db)}")
    
    if len(db) > 0:
        row = db.get(id=1)
        print("\n--- Example Row 1 ---")
        print(f"Formula: {row.formula}")
        print("Keys in row:")
        keys = list(row.keys())
        # Filter out some standard keys to find property keys
        prop_keys = [k for k in keys if k not in ['id', 'unique_id', 'ctime', 'mtime', 'user', 'calculator', 'calculator_parameters', 'cell', 'pbc', 'positions', 'numbers', 'initial_magmoms', 'stress', 'forces', 'energy', 'dipole']]
        print(prop_keys)
        
        print("\n--- Detailed Key Values ---")
        for k in prop_keys[:10]: # Show first 10 properties
            val = getattr(row, k)
            if isinstance(val, (int, float, str)):
                print(f"{k}: {val}")
            else:
                print(f"{k}: {type(val)}")
                
        # Check for specific properties mentioned in proposal
        targets = ['formation_energy', 'band_gap', 'magnetic_moment', 'fermi_level']
        print("\n--- Check Specific Targets ---")
        for t in targets:
            if hasattr(row, t) or t in row.data:
                print(f"Found {t}: Yes")
            else:
                # Try to find similar keys
                similar = [k for k in keys if t in k]
                if similar:
                    print(f"Found similar to {t}: {similar}")
                else:
                    print(f"Found {t}: No")

except Exception as e:
    print(f"Error reading with ASE: {e}")
    # Fallback to sqlite3 if ASE fails
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"\nSQLite Tables: {tables}")
        
        if tables:
            tname = tables[0][0]
            cursor.execute(f"PRAGMA table_info({tname})")
            columns = cursor.fetchall()
            print(f"Columns in {tname}:")
            for col in columns:
                print(col[1])
    except Exception as e2:
        print(f"Error with sqlite3: {e2}")
