from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weather_clean.csv"
)


LABEL_COLUMNS = [
    "temperature_anomaly",
    "humidity_anomaly",
    "pressure_anomaly",
    "any_anomaly",
    "temperature_fault_type",
    "humidity_fault_type",
    "pressure_fault_type",
    "original_temperature_anomaly",
    "original_humidity_anomaly",
    "original_pressure_anomaly",
    "original_any_anomaly",
]


PRECOMPUTED_FEATURES = [
    "T2M_diff",
    "T2M_rolling_mean_6h",
    "T2M_rolling_std_6h",
    "RH2M_diff",
    "RH2M_rolling_mean_6h",
    "RH2M_rolling_std_6h",
    "PS_diff",
    "PS_rolling_mean_6h",
    "PS_rolling_std_6h",
]


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():

    section("ATMOS LEAKAGE AUDIT")

    df = pd.read_csv(INPUT_FILE)

    print(f"Dataset shape: {df.shape}")

    section("1. LABEL COLUMNS")

    labels_present = [
        column
        for column in LABEL_COLUMNS
        if column in df.columns
    ]

    for column in labels_present:
        print(f"[LABEL] {column}")

    section("2. PRECOMPUTED FEATURES")

    features_present = [
        column
        for column in PRECOMPUTED_FEATURES
        if column in df.columns
    ]

    for column in features_present:
        print(f"[REBUILD] {column}")

    section("3. SAFE RAW SENSOR FEATURES")

    safe_features = [
        "T2M",
        "RH2M",
        "PS",
        "T2M_MAX",
        "T2M_MIN",
        "T2M_MEAN",
        "RH2M_MEAN",
        "RH2M_MAX",
        "RH2M_MIN",
        "PS_MEAN",
        "T2M_RANGE",
    ]

    safe_present = [
        column
        for column in safe_features
        if column in df.columns
    ]

    for column in safe_present:
        print(f"[SAFE] {column}")

    section("4. METADATA")

    metadata = [
        "timestamp",
        "date",
        "time",
        "state",
        "station",
        "latitude",
        "longitude",
        "hour",
        "day_of_week",
        "hour_sin",
        "hour_cos",
        "split",
    ]

    for column in metadata:
        if column in df.columns:
            print(f"[META] {column}")

    section("5. FINAL DECISION")

    print(
        """
The ML models will NOT receive anomaly labels or fault labels.

The precomputed temporal features will also be rebuilt
inside our own feature-engineering pipeline.

This prevents the model from learning from information
that would not be available at prediction time.
"""
    )

    print("STATUS: LEAKAGE AUDIT COMPLETE")


if __name__ == "__main__":
    main()