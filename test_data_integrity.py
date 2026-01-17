import json
import time
import math
import sys
import os
try:
    import resource
except ImportError:
    resource = None

def calculate_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def main():
    start_time = time.time()
    
    results = {
        "status": "pending",
        "execution_time_ms": 0,
        "memory_used_mb": 0,
        "total_records": 0,
        "valid_records": 0,
        "invalid_records": 0,
        "errors": [],
        "stats": {
            "avg_atoms": 0,
            "min_target": float('inf'),
            "max_target": float('-inf'),
            "avg_target": 0
        }
    }
    
    print("Starting Data Integrity Test...")
    print("--------------------------------")
    
    try:
        # 1. Load Data
        file_path = 'imp2d_data.json'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found")
            
        print(f"Loading {file_path}...")
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        results["total_records"] = len(data)
        print(f"Loaded {len(data)} records.")
        
        # 2. Validate Records
        valid_count = 0
        total_atoms = 0
        target_sum = 0
        
        print("Validating records...")
        for i, entry in enumerate(data):
            try:
                # Check required keys
                if "jid" not in entry or "atoms" not in entry or "target" not in entry:
                    raise ValueError(f"Record {i} missing keys")
                
                atoms = entry["atoms"]
                if "coords" not in atoms or "lattice_mat" not in atoms or "elements" not in atoms:
                    raise ValueError(f"Record {i} malformed atom data")
                
                # Check dimensions
                num_atoms = len(atoms["elements"])
                if len(atoms["coords"]) != num_atoms:
                    raise ValueError(f"Record {i} mismatch: {num_atoms} elements but {len(atoms['coords'])} coords")
                
                if len(atoms["lattice_mat"]) != 3:
                     raise ValueError(f"Record {i} invalid lattice")

                # Basic Type Checks
                target = entry["target"]
                if not isinstance(target, (int, float)):
                    raise ValueError(f"Record {i} target is not a number: {target}")

                # Update Stats
                total_atoms += num_atoms
                target_sum += target
                results["stats"]["min_target"] = min(results["stats"]["min_target"], target)
                results["stats"]["max_target"] = max(results["stats"]["max_target"], target)
                
                # "Mock" Graph Construction (Check if coordinates are sane)
                # Check distance of first atom to others to ensure no NaN/Inf
                if num_atoms > 1:
                    p0 = atoms["coords"][0]
                    p1 = atoms["coords"][1]
                    dist = calculate_distance(p0, p1)
                    if not math.isfinite(dist):
                         raise ValueError(f"Record {i} infinite distance detected")

                valid_count += 1
                
            except Exception as e:
                if len(results["errors"]) < 10: # limit log size
                    results["errors"].append(str(e))
        
        results["valid_records"] = valid_count
        results["invalid_records"] = results["total_records"] - valid_count
        
        if valid_count > 0:
            results["stats"]["avg_atoms"] = total_atoms / valid_count
            results["stats"]["avg_target"] = target_sum / valid_count
        
        results["status"] = "passed" if results["invalid_records"] == 0 else "warning"
        print(f"Validation complete. Valid: {valid_count}, Invalid: {results['invalid_records']}")

    except Exception as e:
        results["status"] = "failed"
        results["errors"].append(str(e))
        print(f"Test Failed: {e}")

    # Metrics
    end_time = time.time()
    results["execution_time_ms"] = (end_time - start_time) * 1000
    
    if resource:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        results["memory_used_mb"] = usage.ru_maxrss / 1024 / 1024 # MacOS is bytes, Linux is KB? Usually KB on linux, bytes on mac? 
        # On Mac `ru_maxrss` is in bytes. On Linux it is in kilobytes.
        # Let's assume MB output is enough of an estimate.
        if sys.platform == 'darwin':
             results["memory_used_mb"] = usage.ru_maxrss / 1024 / 1024
        else:
             results["memory_used_mb"] = usage.ru_maxrss / 1024

    # Output
    print("\nTest Results Summary")
    print("====================")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
