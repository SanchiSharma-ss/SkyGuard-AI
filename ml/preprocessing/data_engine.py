from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# ============================================================
# ATMOS - DATA ENGINE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = RAW_DIR / "weather_complete_anomaly_dataset01.csv"
OUTPUT_FILE = PROCESSED_DIR / "weather_clean.csv"
REPORT_FILE = PROCESSED_DIR / "data_quality_report.json"


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_dataset():
    print_section("1. LOADING DATASET")

    if not INPUT_FILE.exists():
        print(f"ERROR: Dataset not found:")
        print(INPUT_FILE)
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE)

    print(f"Dataset loaded successfully.")
    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")

    return df


def inspect_schema(df):
    print_section("2. SCHEMA INSPECTION")

    print("Columns:")
    for i, column in enumerate(df.columns, start=1):
        print(f"{i:02d}. {column}")

    print("\nData types:")
    print(df.dtypes.to_string())


def inspect_missing_values(df):
    print_section("3. MISSING VALUE ANALYSIS")

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("No missing values detected.")
    else:
        print(missing.to_string())

    return missing


def inspect_duplicates(df):
    print_section("4. DUPLICATE ANALYSIS")

    duplicate_count = int(df.duplicated().sum())

    print(f"Duplicate rows: {duplicate_count:,}")

    return duplicate_count


def inspect_numeric_data(df):
    print_section("5. NUMERIC DATA ANALYSIS")

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    print(f"Numeric columns: {len(numeric_columns)}")

    if numeric_columns:
        print("\nNumeric summary:")
        print(
            df[numeric_columns]
            .describe()
            .transpose()
            .to_string()
        )

    return numeric_columns


def create_timestamp(df):
    print_section("6. TIMESTAMP PROCESSING")

    date_column = None
    time_column = None

    for column in df.columns:
        normalized = column.strip().lower()

        if normalized == "date":
            date_column = column

        elif normalized == "time":
            time_column = column

    if date_column and time_column:

        print(
            f"Combining '{date_column}' + '{time_column}' "
            "into timestamp..."
        )

        df["timestamp"] = pd.to_datetime(
            df[date_column].astype(str)
            + " "
            + df[time_column].astype(str),
            errors="coerce"
        )

        invalid_timestamp_count = int(
            df["timestamp"].isna().sum()
        )

        print(
            f"Invalid timestamps: "
            f"{invalid_timestamp_count:,}"
        )

        return df

    print(
        "Date/time columns were not found. "
        "Timestamp was not created."
    )

    return df


def clean_dataset(df):
    print_section("7. DATA CLEANING")

    original_rows = len(df)

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    # Sort chronologically if timestamp exists
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    df = df.reset_index(drop=True)

    removed_rows = original_rows - len(df)

    print(f"Original rows : {original_rows:,}")
    print(f"Final rows    : {len(df):,}")
    print(f"Removed rows  : {removed_rows:,}")

    return df, removed_rows


def generate_report(
    df_original,
    df_clean,
    missing,
    duplicate_count,
    removed_rows,
    numeric_columns
):
    print_section("8. GENERATING DATA QUALITY REPORT")

    report = {
        "project": "ATMOS",
        "dataset": INPUT_FILE.name,

        "original_shape": {
            "rows": int(df_original.shape[0]),
            "columns": int(df_original.shape[1])
        },

        "cleaned_shape": {
            "rows": int(df_clean.shape[0]),
            "columns": int(df_clean.shape[1])
        },

        "removed_rows": int(removed_rows),

        "duplicate_rows_detected": int(duplicate_count),

        "missing_values": {
            str(column): int(value)
            for column, value in missing.items()
        },

        "numeric_columns": numeric_columns,

        "columns": [
            str(column)
            for column in df_clean.columns
        ]
    }

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print(f"Report saved to:")
    print(REPORT_FILE)


def save_clean_dataset(df):
    print_section("9. SAVING CLEAN DATASET")

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("Clean dataset saved:")
    print(OUTPUT_FILE)

    print(
        f"Final dataset shape: "
        f"{df.shape[0]:,} × {df.shape[1]}"
    )


def main():

    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                    ATMOS DATA ENGINE                        ║")
    print("║        Automatic Weather Station Data Pipeline              ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    df_original = load_dataset()

    inspect_schema(df_original)

    missing = inspect_missing_values(
        df_original
    )

    duplicate_count = inspect_duplicates(
        df_original
    )

    numeric_columns = inspect_numeric_data(
        df_original
    )

    df = create_timestamp(
        df_original.copy()
    )

    df_clean, removed_rows = clean_dataset(
        df
    )

    generate_report(
        df_original=df_original,
        df_clean=df_clean,
        missing=missing,
        duplicate_count=duplicate_count,
        removed_rows=removed_rows,
        numeric_columns=numeric_columns
    )

    save_clean_dataset(
        df_clean
    )

    print_section("10. DATA ENGINE COMPLETE")

    print("STATUS: SUCCESS")
    print()
    print("Generated:")
    print(f"1. {OUTPUT_FILE}")
    print(f"2. {REPORT_FILE}")
    print()


if __name__ == "__main__":
    main()