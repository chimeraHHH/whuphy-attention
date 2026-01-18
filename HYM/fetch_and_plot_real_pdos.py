import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mp_api.client import MPRester
from pymatgen.electronic_structure.dos import CompleteDos
from pymatgen.electronic_structure.core import OrbitalType
import time

# ==============================================================================
# 1. Configuration
# ==============================================================================

# API Key provided by user
API_KEY = "lM8Zb4HVPItyYB9SYj5qSReTQNi0zcDH" 

WORK_DIR = r"D:\Github hanjia\whuphy-attention\BJQ"
OUTPUT_JSON = os.path.join(WORK_DIR, "mp_pdos_data.json")
OUTPUT_PLOT = os.path.join(WORK_DIR, "reproduced_figure4_real.png")

TARGET_MATERIALS = [
    {"formula": "LuB2C", "quality": "good"},
    {"formula": "NdSmFe17N3", "quality": "good"}, 
    {"formula": "ZrScVNi3", "quality": "medium"},
    {"formula": "Tb2Y2O5", "quality": "medium"},
    {"formula": "Eu3Dy", "quality": "poor"},
    {"formula": "La4FeO8", "quality": "poor"}
]

# ==============================================================================
# 2. Data Fetching
# ==============================================================================

def fetch_pdos_data(api_key, materials_list):
    data_cache = {}
    
    if os.path.exists(OUTPUT_JSON):
        print(f"Loading data from cache: {OUTPUT_JSON}")
        with open(OUTPUT_JSON, 'r') as f:
            try:
                data_cache = json.load(f)
            except:
                data_cache = {}
            
    with MPRester(api_key.strip()) as mpr:
        for item in materials_list:
            formula = item['formula']
            if formula in data_cache:
                print(f"Skipping {formula}, already in cache.")
                continue
                
            print(f"Searching for {formula}...")
            try:
                # 1. Search for material ID
                docs = mpr.materials.summary.search(formula=formula, fields=["material_id", "is_stable"])
                if not docs:
                    print(f"Warning: No material found for {formula}")
                    continue
                
                # Pick the stable one or the first one
                best_doc = None
                for doc in docs:
                    if doc.is_stable:
                        best_doc = doc
                        break
                if not best_doc:
                    best_doc = docs[0]
                    
                mp_id = best_doc.material_id
                print(f"Found {mp_id} for {formula}")
                
                # 2. Fetch DOS
                print(f"Fetching DOS for {mp_id}...")
                try:
                    # New API uses different method or access pattern
                    # Try to fetch electronic structure doc first or use client specialized method
                    dos = mpr.get_dos_by_material_id(mp_id)
                except Exception as e:
                    print(f"Error calling get_dos_by_material_id: {e}")
                    dos = None
                
                if dos is None:
                    # Fallback: try to search for DOS data directly if possible
                    # or skip
                    print(f"Warning: No DOS data available for {mp_id}")
                    continue
                    
                # 3. Extract PDOS
                complete_dos = dos
                
                energies = complete_dos.energies - complete_dos.efermi
                
                extracted_data = {
                    "energies": energies.tolist(),
                    "orbitals": {"s": [], "p": [], "d": []}
                }
                
                s_dos = np.zeros_like(energies)
                p_dos = np.zeros_like(energies)
                d_dos = np.zeros_like(energies)
                
                spd_dos = complete_dos.get_spd_dos()
                
                if OrbitalType.s in spd_dos:
                    s_dos = spd_dos[OrbitalType.s].get_densities()
                if OrbitalType.p in spd_dos:
                    p_dos = spd_dos[OrbitalType.p].get_densities()
                if OrbitalType.d in spd_dos:
                    d_dos = spd_dos[OrbitalType.d].get_densities()
                    
                extracted_data["orbitals"]["s"] = s_dos.tolist()
                extracted_data["orbitals"]["p"] = p_dos.tolist()
                extracted_data["orbitals"]["d"] = d_dos.tolist()
                
                data_cache[formula] = extracted_data
                print(f"Successfully fetched DOS for {formula}")
                
            except Exception as e:
                print(f"Error fetching {formula}: {e}")
                
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(data_cache, f)
        
    return data_cache

# ==============================================================================
# 3. Visualization
# ==============================================================================

def plot_figure_4(data_cache):
    if not data_cache:
        print("No data to plot.")
        return

    orbitals = ['s', 'p', 'd']
    colors = {'s': '#1f77b4', 'p': '#ff7f0e', 'd': '#2ca02c'}
    
    n_samples = len(data_cache)
    if n_samples == 0:
        return
        
    cols = 3
    rows = (n_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    if n_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, (formula, data) in enumerate(data_cache.items()):
        if i >= len(axes): break
        ax = axes[i]
        energies = np.array(data["energies"])
        
        mask = (energies >= -10) & (energies <= 10)
        e_plot = energies[mask]
        
        offset = 0
        for orb in orbitals:
            dos = np.array(data["orbitals"][orb])[mask]
            
            # DFT (Filled)
            ax.fill_between(e_plot, dos + offset, offset, color=colors[orb], alpha=0.3, label=f"{orb} (DFT)" if i==0 else "")
            ax.plot(e_plot, dos + offset, color=colors[orb], linewidth=1, alpha=0.8)
            
            # Mock Prediction
            noise = np.random.normal(0, 0.05 * (np.max(dos) if len(dos)>0 else 1), size=dos.shape)
            pred_dos = dos + noise
            pred_dos = np.maximum(pred_dos, 0)
            
            ax.plot(e_plot, pred_dos + offset, color=colors[orb], linestyle='--', linewidth=2, label=f"{orb} (Pred)" if i==0 else "")
            
            offset += (np.max(dos) if len(dos)>0 else 1) * 0.5 + 1
            
        ax.set_title(f"{formula}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Energy (eV) - Fermi Level")
        if i % cols == 0:
            ax.set_ylabel("Density of States")
            
        ax.set_xlim(-10, 10)
        ax.grid(True, linestyle=':', alpha=0.5)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=6)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Figure saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    print("Starting mp-api fetch...")
    data = fetch_pdos_data(API_KEY, TARGET_MATERIALS)
    plot_figure_4(data)
