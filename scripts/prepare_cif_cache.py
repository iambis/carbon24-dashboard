"""Cache Carbon-24 CIF strings for the Streamlit 3D structure viewer.

Run once:

    python prepare_cif_cache.py

Output:
    carbon24_cif_cache.csv
"""

from pathlib import Path

import pandas as pd


OUTPUT_PATH = Path("carbon24_cif_cache.csv")


def main():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install with `pip install datasets`."
        ) from exc

    ds = load_dataset("albertvillanova/carbon_24")
    rows = []

    for split in ds.keys():
        part = ds[split].to_pandas()
        part["split"] = split

        required = {"material_id", "cif", "energy_per_atom", "split"}
        missing = required - set(part.columns)
        if missing:
            raise ValueError(f"Dataset split {split} missing columns: {sorted(missing)}")

        rows.append(part[["material_id", "cif", "energy_per_atom", "split"]])

    out = pd.concat(rows, ignore_index=True).drop_duplicates("material_id")
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(out):,} CIF records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
