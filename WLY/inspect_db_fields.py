import argparse
import json
import os
import sqlite3
from pathlib import Path


def inspect_ase_db(db_path: Path) -> None:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    print(f"Inspecting DB: {db_path}")
    print("-" * 60)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    print("[systems columns]")
    cursor.execute("PRAGMA table_info(systems)")
    columns = cursor.fetchall()
    for col in columns:
        name = col[1]
        ctype = col[2]
        cursor.execute(f"SELECT 1 FROM systems WHERE {name} IS NOT NULL LIMIT 1")
        has_data = cursor.fetchone() is not None
        status = "non-null" if has_data else "all-null"
        print(f"  - {name:<25} ({ctype:<8}) {status}")

    print("\n[key-value fields]")
    metadata_keys: set[tuple[str, str]] = set()
    for table, dtype in [("text_key_values", "TEXT"), ("number_key_values", "NUMBER")]:
        try:
            cursor.execute(f"SELECT DISTINCT key FROM {table}")
            for (key,) in cursor.fetchall():
                metadata_keys.add((key, dtype))
        except sqlite3.OperationalError:
            pass

    if not metadata_keys:
        cursor.execute("SELECT key_value_pairs FROM systems LIMIT 200")
        for (raw,) in cursor.fetchall():
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            for key, value in data.items():
                dtype = "NUMBER" if isinstance(value, (int, float)) else "TEXT"
                metadata_keys.add((key, dtype))

    for key, dtype in sorted(metadata_keys):
        print(f"  - {key:<30} ({dtype})")

    cursor.execute("SELECT COUNT(*) FROM systems")
    total = cursor.fetchone()[0]
    print("-" * 60)
    print(f"Rows in systems: {total}")
    print(f"Metadata key count: {len(metadata_keys)}")
    conn.close()


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


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Inspect ASE sqlite fields and metadata keys")
    parser.add_argument("--db", default=str(resolve_default_db(base_dir)))
    args = parser.parse_args()
    inspect_ase_db(Path(args.db))


if __name__ == "__main__":
    main()
