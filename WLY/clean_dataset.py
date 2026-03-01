import argparse
import json
import os
import pickle
import sqlite3
from pathlib import Path

import numpy as np


def resolve_default_db(base_dir: Path) -> Path:
    candidates = [
        base_dir / "imp2d.db",
        base_dir.parent / "imp2d.db",
        base_dir.parent / "BJQ" / "imp2d.db",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def decode_pbc(pbc_int: int) -> np.ndarray:
    # ASE stores PBC flags as bit mask in some sqlite dumps.
    return np.array([(pbc_int >> i) & 1 for i in range(3)], dtype=bool)


def clean_data(db_path: Path, output_file: Path) -> None:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    print(f"Processing DB: {db_path}")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, unique_id, numbers, positions, cell, pbc, key_value_pairs
        FROM systems
        """
    )

    cleaned_samples = []
    total_count = 0
    kept_count = 0

    for row in cursor:
        total_count += 1
        row_id, unique_id, numbers_blob, positions_blob, cell_blob, pbc_int, kv_json = row

        if not kv_json:
            continue

        try:
            kv = json.loads(kv_json)
        except json.JSONDecodeError:
            continue

        converged = kv.get("converged")
        if converged is not True and converged != 1:
            continue

        eform = kv.get("eform")
        if eform is None:
            continue

        try:
            eform_val = float(eform)
            if np.isnan(eform_val):
                continue
        except (TypeError, ValueError):
            continue

        try:
            numbers = np.frombuffer(numbers_blob, dtype=np.int32)
            positions = np.frombuffer(positions_blob, dtype=np.float64).reshape(-1, 3)
            cell = np.frombuffer(cell_blob, dtype=np.float64).reshape(3, 3)
            pbc = decode_pbc(int(pbc_int))
        except Exception:
            continue

        sample = {
            "id": row_id,
            "unique_id": unique_id,
            "numbers": numbers,
            "positions": positions,
            "cell": cell,
            "pbc": pbc,
            "target": eform_val,
            "metadata": {
                "formula": kv.get("name", ""),
                "host": kv.get("host"),
                "dopant": kv.get("dopant"),
                "site": kv.get("site"),
                "defecttype": kv.get("defecttype"),
                "natoms": len(numbers),
            },
        }
        cleaned_samples.append(sample)
        kept_count += 1

    conn.close()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as f:
        pickle.dump(cleaned_samples, f)

    print("-" * 50)
    print(f"Total rows: {total_count}")
    print(f"Kept rows: {kept_count}")
    print(f"Dropped rows: {total_count - kept_count}")
    print(f"Saved: {output_file}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    default_db = resolve_default_db(base_dir)
    default_output = base_dir / "cleaned_dataset.pkl"

    parser = argparse.ArgumentParser(description="Clean imp2d ASE sqlite dataset")
    parser.add_argument("--db", default=str(default_db), help="Path to imp2d.db")
    parser.add_argument(
        "--output",
        default=str(default_output),
        help="Output pickle file (cleaned dataset)",
    )
    args = parser.parse_args()

    clean_data(Path(args.db), Path(args.output))


if __name__ == "__main__":
    main()
