import pickle
import numpy as np
import os

INPUT_PATH = '/Users/wuleyan/Desktop/dachuang/whuphy-attention/WLY/processed_dataset_with_graphs.pkl'
OUTPUT_PATH = '/Users/wuleyan/Desktop/dachuang/whuphy-attention/WLY/final_dataset.pkl'

def filter_outliers():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found.")
        return

    print(f"Loading dataset from {INPUT_PATH}...")
    with open(INPUT_PATH, 'rb') as f:
        data = pickle.load(f)

    print(f"Original dataset size: {len(data)}")

    # Define reasonable range for formation energy (eV)
    # Based on percentiles: 1% is -170 (still likely bad), 5% is -1.97.
    # 99% is 19.81.
    # Let's be slightly conservative but remove the massive outliers.
    # Range [-20, 20] eV covers >95% of data and removes the -800k error.
    MIN_VAL = -20.0
    MAX_VAL = 20.0
    
    filtered_data = []
    rejected_count = 0
    
    targets = []
    
    for sample in data:
        t = sample['target']
        if MIN_VAL <= t <= MAX_VAL:
            filtered_data.append(sample)
            targets.append(t)
        else:
            rejected_count += 1

    print(f"Filtering criteria: {MIN_VAL} <= target <= {MAX_VAL}")
    print(f"Kept samples: {len(filtered_data)}")
    print(f"Rejected outliers: {rejected_count}")
    
    # New stats
    if len(targets) > 0:
        targets = np.array(targets)
        print(f"New Mean: {targets.mean():.4f}")
        print(f"New Std:  {targets.std():.4f}")
        print(f"New Min:  {targets.min():.4f}")
        print(f"New Max:  {targets.max():.4f}")

    print(f"Saving cleaned dataset to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(filtered_data, f)
        
    print("Done!")

if __name__ == "__main__":
    filter_outliers()
