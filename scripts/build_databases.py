"""
build_databases.py
-------------------
Converts the Bangladesh datasets (CSV) into three SQLite databases with
clean, meaningful column names and correct types:

    data/db/institutions.db  -> table: institutions
    data/db/hospitals.db     -> table: hospitals
    data/db/restaurants.db   -> table: restaurants

By default it builds from the small bundled sample under data/sample/ so the
project runs end-to-end with zero setup. Once you've run
scripts/download_datasets.py, rebuild from the full data with:

    python scripts/build_databases.py --source raw
"""
import argparse
import os
import sqlite3

import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
SAMPLE_DIR = os.path.join(BASE_DIR, "data", "sample")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DB_DIR = os.path.join(BASE_DIR, "data", "db")


# ---------------------------------------------------------------------------
# Column rename maps: raw HuggingFace column -> clean, query-friendly column.
# Renaming happens BEFORE type inference so downstream tools can rely on
# stable, human-readable names (e.g. "district" instead of "DISTRICT").
# ---------------------------------------------------------------------------

INSTITUTIONS_RENAME = {
    "INSTITUTE_NAME": "name",
    "EIIN": "eiin_code",
    "INSTITUTE_TYPE": "institute_type",
    "DIVISION_ID": "division_id",
    "DIVISION": "division",
    "DISTRICT_ID": "district_id",
    "DISTRICT": "district",
    "THANA_ID": "thana_id",
    "THANA": "thana",
    "UNION_ID": "union_id",
    "UNION_NAME": "union_name",
    "MAUZA_ID": "mauza_id",
    "MAUZA_NAME": "mauza_name",
    "AREA_STATUS": "area_status",
    "GEOGRPYCAL_STATUS": "geographical_status",  # raw dataset has this typo
    "GEOGRAPHICAL_STATUS": "geographical_status",
    "ADDRESS": "address",
    "POST": "post_office",
    "MANAGEMENT_TYPE": "management_type",
    "MOBILE": "mobile",
    "STUDENT_TYPE": "student_type",
    "EDUCATION_LEVEL": "education_level",
    "AFFILIATION": "affiliation_status",
    "MPO_STATUS": "mpo_status",
}

HOSPITALS_RENAME = {
    "Id": "facility_id",
    "Name": "name",
    "Name_Bangla": "name_bangla",
    "Name (Bangla)": "name_bangla",
    "Code": "facility_code",
    "Agency": "agency",
    "Type": "facility_type",
    "Division": "division",
    "District": "district",
    "City_Corporation": "city_corporation",
    "City Corporation": "city_corporation",
    "Upazila": "upazila",
    "Paurasava": "paurasava",
    "Union": "union_name",
    "Private": "is_private",
}

RESTAURANTS_RENAME = {
    "place_id": "place_id",
    "name": "name",
    "latitude": "latitude",
    "longitude": "longitude",
    "rating": "rating",
    "number_of_reviews": "number_of_reviews",
    "affluence": "affluence",
    "address": "address",
}

# A small keyword list used to derive a best-effort `cuisine_guess` column
# for restaurants, since the source dataset has no explicit cuisine field.
CUISINE_KEYWORDS = {
    "biryani": ["biryani", "biriyani", "birayani", "kacchi"],
    "chinese": ["chinese", "china"],
    "fast_food": ["fast food", "burger", "pizza", "kfc"],
    "sweets_bakery": ["sweet", "misti", "bakery", "confectionary", "cake"],
    "tea_stall": ["tea", "cha "],
    "traditional_bangladeshi": ["hotel", "restaurant", "restaurent", "bhater", "panshi"],
    "cafe": ["cafe", "coffee"],
}


def guess_cuisine(name: str) -> str:
    if not isinstance(name, str):
        return "unknown"
    lowered = name.lower()
    for cuisine, keywords in CUISINE_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return cuisine
    return "unknown"


def infer_sqlite_type(series: pd.Series) -> str:
    """Map a pandas dtype to a SQLite column type (TEXT / INTEGER / REAL)."""
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series):
        return "REAL"
    return "TEXT"


def write_table(df: pd.DataFrame, db_path: str, table_name: str) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Build an explicit CREATE TABLE statement so column types are intentional
    # rather than left to SQLite's dynamic typing / pandas' default inference.
    col_defs = []
    for col in df.columns:
        sqlite_type = infer_sqlite_type(df[col])
        safe_col = col.strip().lower().replace(" ", "_")
        col_defs.append(f'"{safe_col}" {sqlite_type}')
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    cur.execute(f'CREATE TABLE "{table_name}" ({", ".join(col_defs)})')
    conn.commit()

    df.to_sql(table_name, conn, if_exists="append", index=False)

    # Helpful indexes for the columns agents will filter on most often.
    for candidate in ("district", "division", "city_corporation", "name"):
        if candidate in df.columns:
            cur.execute(
                f'CREATE INDEX IF NOT EXISTS idx_{table_name}_{candidate} '
                f'ON "{table_name}" ("{candidate}")'
            )
    conn.commit()
    conn.close()
    print(f"  -> {db_path} :: table '{table_name}' ({len(df):,} rows, {len(df.columns)} cols)")


def build_institutions(source_dir: str) -> None:
    path = os.path.join(source_dir, "institutions_raw.csv" if source_dir == RAW_DIR else "institutions_sample.csv")
    print(f"[institutions] reading {path}")
    df = pd.read_csv(path)
    df = df.rename(columns=INSTITUTIONS_RENAME)
    write_table(df, os.path.join(DB_DIR, "institutions.db"), "institutions")


def build_hospitals(source_dir: str) -> None:
    path = os.path.join(source_dir, "hospitals_raw.csv" if source_dir == RAW_DIR else "hospitals_sample.csv")
    print(f"[hospitals] reading {path}")
    df = pd.read_csv(path)
    df = df.rename(columns=HOSPITALS_RENAME)
    if "is_private" in df.columns:
        df["is_private"] = pd.to_numeric(df["is_private"], errors="coerce").fillna(0).astype(int)
    write_table(df, os.path.join(DB_DIR, "hospitals.db"), "hospitals")


def build_restaurants(source_dir: str) -> None:
    path = os.path.join(source_dir, "restaurants_raw.csv" if source_dir == RAW_DIR else "restaurants_sample.csv")
    print(f"[restaurants] reading {path}")
    df = pd.read_csv(path)
    df = df.rename(columns=RESTAURANTS_RENAME)
    for col in ("rating", "number_of_reviews", "latitude", "longitude", "affluence"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["cuisine_guess"] = df["name"].apply(guess_cuisine)
    write_table(df, os.path.join(DB_DIR, "restaurants.db"), "restaurants")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["sample", "raw"],
        default="sample",
        help="Build from the bundled sample data (default) or the full "
        "downloaded data in data/raw/ (run scripts/download_datasets.py first).",
    )
    args = parser.parse_args()
    source_dir = RAW_DIR if args.source == "raw" else SAMPLE_DIR

    print(f"Building SQLite databases from '{args.source}' data...\n")
    build_institutions(source_dir)
    build_hospitals(source_dir)
    build_restaurants(source_dir)
    print("\nAll databases built in data/db/")


if __name__ == "__main__":
    main()
