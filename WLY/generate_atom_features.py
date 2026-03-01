import argparse
import json
from pathlib import Path

import numpy as np
import torch
from mendeleev import element


MAX_Z = 100


def build_features(max_z: int) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    feature_names = [
        "Group",
        "Period",
        "Electronegativity",
        "CovalentRadius",
        "VdWRadius",
        "ValenceElectrons",
        "IonizationEnergy",
        "ElectronAffinity",
        "AtomicMass",
    ]

    rows = [[0.0] * len(feature_names)]  # Padding at index 0

    for z in range(1, max_z + 1):
        try:
            el = element(z)
            group = el.group_id if el.group_id is not None else 0
            period = el.period if el.period is not None else 0
            en = el.electronegativity("pauling") or 0.0
            rcov = el.covalent_radius or 0.0
            rvdw = el.vdw_radius or 0.0

            n_valence = 0
            try:
                val = el.nvalence()
                if val is not None:
                    n_valence = val
            except Exception:
                n_valence = 0

            ion_energy = el.ionenergies.get(1, 0.0) if el.ionenergies else 0.0
            ea = el.electron_affinity or 0.0
            mass = el.atomic_weight if el.atomic_weight is not None else 0.0

            feat = [
                float(group),
                float(period),
                float(en),
                float(rcov),
                float(rvdw),
                float(n_valence),
                float(ion_energy),
                float(ea),
                float(mass),
            ]
            rows.append(feat)
        except Exception:
            rows.append([0.0] * len(feature_names))

    raw = np.array(rows, dtype=np.float32)
    data = raw[1:]
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    normalized = (data - min_vals) / ranges
    final = np.vstack([raw[0], normalized]).astype(np.float32)
    return final, feature_names, min_vals, max_vals


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate normalized atom feature table")
    parser.add_argument("--output", default=str(base_dir / "atom_features.pth"))
    parser.add_argument("--json_output", default=str(base_dir / "atom_features.json"))
    parser.add_argument("--max_z", type=int, default=MAX_Z)
    args = parser.parse_args()

    features, feature_names, min_vals, max_vals = build_features(args.max_z)

    output_path = Path(args.output)
    json_output_path = Path(args.json_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(torch.from_numpy(features), output_path)

    payload = {
        "feature_names": feature_names,
        "normalization_params": {"min": min_vals.tolist(), "max": max_vals.tolist()},
        "data_normalized": features.tolist(),
    }
    with json_output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved tensor: {output_path}")
    print(f"Saved json: {json_output_path}")


if __name__ == "__main__":
    main()
