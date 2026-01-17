import sqlite3
import json
import struct
import math

# Minimal Periodic Table for Z -> Symbol mapping
ELEMENTS = [
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]

def get_symbol(z):
    if 0 < z < len(ELEMENTS):
        return ELEMENTS[z]
    return f"X{z}"

def decode_array(blob, dtype='d'):
    try:
        if not blob: return []
        element_size = struct.calcsize(dtype)
        count = len(blob) // element_size
        fmt = f"{count}{dtype}"
        return list(struct.unpack(fmt, blob))
    except:
        return []

def main():
    db_path = 'imp2d.db'
    output_path = 'imp2d_data.json'
    log_path = 'import_log.txt'
    
    print(f"Opening database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check columns
    cursor.execute("PRAGMA table_info(systems)")
    columns = [col[1] for col in cursor.fetchall()]
    
    # Select necessary columns
    # We need: unique_id, numbers, positions, cell, key_value_pairs
    query = "SELECT unique_id, numbers, positions, cell, key_value_pairs FROM systems"
    cursor.execute(query)
    
    valid_data = []
    total_count = 0
    skipped_count = 0
    skipped_reasons = {"converged": 0, "no_eform": 0, "decode_error": 0}
    
    for row in cursor:
        total_count += 1
        uid, numbers_blob, pos_blob, cell_blob, kv_json = row
        
        try:
            # Parse metadata
            kv = json.loads(kv_json)
            
            # Check convergence
            if kv.get("converged") is False:
                skipped_count += 1
                skipped_reasons["converged"] += 1
                continue
                
            # Check target (eform)
            eform = kv.get("eform")
            if eform is None or math.isnan(eform):
                skipped_count += 1
                skipped_reasons["no_eform"] += 1
                continue
            
            # Decode structure
            # Atomic numbers: try int32 ('i') first based on previous inspection
            numbers = decode_array(numbers_blob, 'i')
            if not numbers: # fallback
                 numbers = decode_array(numbers_blob, 'q')

            positions_flat = decode_array(pos_blob, 'd')
            cell_flat = decode_array(cell_blob, 'd')
            
            num_atoms = len(numbers)
            
            # Reshape positions (N x 3)
            coords = []
            for i in range(num_atoms):
                coords.append(positions_flat[i*3 : (i+1)*3])
            
            # Reshape cell (3 x 3)
            lattice_mat = []
            for i in range(3):
                lattice_mat.append(cell_flat[i*3 : (i+1)*3])
            
            # Convert Z to symbols
            elements = [get_symbol(z) for z in numbers]
            
            # Construct dictionary
            entry = {
                "jid": uid,
                "atoms": {
                    "lattice_mat": lattice_mat,
                    "coords": coords,
                    "elements": elements,
                    "cartesian": True
                },
                "target": eform,
                "extra_info": kv 
            }
            valid_data.append(entry)
            
        except Exception as e:
            skipped_count += 1
            skipped_reasons["decode_error"] += 1
            # print(f"Error processing row {uid}: {e}")

    conn.close()
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(valid_data, f, indent=2)
        
    # Generate log
    log_content = f"""Import Report
================================
Total records processed: {total_count}
Successfully imported: {len(valid_data)}
Skipped: {skipped_count}
  - Not Converged: {skipped_reasons['converged']}
  - No Formation Energy (eform): {skipped_reasons['no_eform']}
  - Decode Errors: {skipped_reasons['decode_error']}

Output file: {output_path}
"""
    with open(log_path, 'w') as f:
        f.write(log_content)
        
    print(log_content)

if __name__ == "__main__":
    main()
