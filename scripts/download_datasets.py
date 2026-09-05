"""
download_datasets.py
---------------------
Downloads the three Bangladesh datasets from Hugging Face and saves them as
CSV files under data/raw/. Requires internet access to huggingface.co.

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --only hospitals
"""
import argparse
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# HuggingFace auto-converts every CSV dataset to a queryable parquet file.
# Reading that parquet file with pandas is the fastest, most reliable way
# to pull the full dataset without needing the `datasets` library.
DATASETS = {
    "institutions": {
        "repo": "Mahadih534/Institutional-Information-of-Bangladesh",
        "parquet": "hf://datasets/Mahadih534/Institutional-Information-of-Bangladesh/default/train-00000-of-00001.parquet",
        "csv_out": "institutions_raw.csv",
    },
    "hospitals": {
        "repo": "Mahadih534/all-bangladeshi-hospitals",
        "parquet": "hf://datasets/Mahadih534/all-bangladeshi-hospitals/default/train-00000-of-00001.parquet",
        "csv_out": "hospitals_raw.csv",
    },
    "restaurants": {
        "repo": "Mahadih534/Bangladeshi-Restaurant-Data",
        "parquet": "hf://datasets/Mahadih534/Bangladeshi-Restaurant-Data/default/train-00000-of-00001.parquet",
        "csv_out": "restaurants_raw.csv",
    },
}


def download_one(key: str, info: dict) -> None:
    print(f"[download] {key}: {info['repo']}")
    try:
        # Preferred path: pandas + huggingface_hub can read the hf:// URI directly
        # (pip install huggingface_hub pandas pyarrow fsspec)
        df = pd.read_parquet(info["parquet"])
    except Exception as e:
        print(f"  parquet read failed ({e}); falling back to `datasets` library...")
        from datasets import load_dataset

        ds = load_dataset(info["repo"], split="train")
        df = ds.to_pandas()

    os.makedirs(RAW_DIR, exist_ok=True)
    out_path = os.path.join(RAW_DIR, info["csv_out"])
    df.to_csv(out_path, index=False)
    print(f"  saved {len(df):,} rows -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=list(DATASETS.keys()),
        default=None,
        help="Download only one dataset instead of all three.",
    )
    args = parser.parse_args()

    targets = {args.only: DATASETS[args.only]} if args.only else DATASETS
    for key, info in targets.items():
        download_one(key, info)

    print("\nDone. Next step: python scripts/build_databases.py --source raw")


if __name__ == "__main__":
    main()
