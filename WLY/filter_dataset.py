import pickle
import numpy as np
import os
import argparse

def filter_outliers(input_path, output_path, min_val, max_val):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    print(f"Loading dataset from {input_path}...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Original dataset size: {len(data)}")

    # Define reasonable range for formation energy (eV)
    # Based on percentiles: 1% is -170 (still likely bad), 5% is -1.97.
    # 99% is 19.81.
    # Let's be slightly conservative but remove the massive outliers.
    # Range [-20, 20] eV covers >95% of data and removes the -800k error.
    filtered_data = []
    rejected_count = 0
    
    targets = []
    
    for sample in data:
        t = sample['target']
        if min_val <= t <= max_val:
            filtered_data.append(sample)
            targets.append(t)
        else:
            rejected_count += 1

    print(f"Filtering criteria: {min_val} <= target <= {max_val}")
    print(f"Kept samples: {len(filtered_data)}")
    print(f"Rejected outliers: {rejected_count}")
    
    # New stats
    if len(targets) > 0:
        targets = np.array(targets)
        print(f"New Mean: {targets.mean():.4f}")
        print(f"New Std:  {targets.std():.4f}")
        print(f"New Min:  {targets.min():.4f}")
        print(f"New Max:  {targets.max():.4f}")

    print(f"Saving cleaned dataset to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(filtered_data, f)
        
    print("Done!")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(base_dir, "processed_dataset_with_graphs.pkl"))
    parser.add_argument("--output", default=os.path.join(base_dir, "final_dataset.pkl"))
    parser.add_argument("--min", type=float, default=-20.0)
    parser.add_argument("--max", type=float, default=20.0)
    args = parser.parse_args()
    filter_outliers(args.input, args.output, args.min, args.max)
