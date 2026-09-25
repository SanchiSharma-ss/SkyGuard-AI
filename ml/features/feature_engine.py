from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weather_clean.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "ml_features.csv"
)


SENSORS = [
    "T2M",
    "RH2M",
    "PS"
]


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_data():

    section("1. LOADING DATA")

    df = pd.read_csv(
        INPUT_FILE
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df = df.sort_values(
        ["station", "timestamp"]
    ).reset_index(drop=True)

    print(
        f"Loaded {len(df):,} records"
    )

    return df


def create_temporal_features(df):

    section("2. TEMPORAL FEATURES")

    timestamp = df["timestamp"]

    df["hour"] = timestamp.dt.hour

    df["day_of_week"] = (
        timestamp.dt.dayofweek
    )

    df["day_of_year"] = (
        timestamp.dt.dayofyear
    )

    df["month"] = (
        timestamp.dt.month
    )

    # Cyclic encoding

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    return df


def create_sensor_features(df):

    section("3. SENSOR TEMPORAL FEATURES")

    grouped = df.groupby(
        "station",
        group_keys=False
    )

    for sensor in SENSORS:

        # Change from previous observation
        df[f"{sensor}_diff"] = (
            grouped[sensor]
            .diff()
        )

        # Percentage change
        previous = (
            grouped[sensor]
            .shift(1)
        )

        df[f"{sensor}_pct_change"] = (
            df[f"{sensor}_diff"]
            / previous.replace(
                0,
                np.nan
            )
        )

        # Previous observation
        df[f"{sensor}_lag_1"] = (
            grouped[sensor]
            .shift(1)
        )

        # Previous 3 observations
        df[f"{sensor}_lag_3"] = (
            grouped[sensor]
            .shift(3)
        )

        # Rolling statistics using past values
        shifted = (
            grouped[sensor]
            .shift(1)
        )

        df[f"{sensor}_rolling_mean_6"] = (
            shifted
            .groupby(df["station"])
            .rolling(
                window=6,
                min_periods=2
            )
            .mean()
            .reset_index(
                level=0,
                drop=True
            )
        )

        df[f"{sensor}_rolling_std_6"] = (
            shifted
            .groupby(df["station"])
            .rolling(
                window=6,
                min_periods=2
            )
            .std()
            .reset_index(
                level=0,
                drop=True
            )
        )

    return df


def create_deviation_features(df):

    section("4. LOCAL DEVIATION FEATURES")

    for sensor in SENSORS:

        mean_column = (
            f"{sensor}_rolling_mean_6"
        )

        std_column = (
            f"{sensor}_rolling_std_6"
        )

        df[f"{sensor}_local_z"] = (
            (
                df[sensor]
                - df[mean_column]
            )
            /
            df[std_column].replace(
                0,
                np.nan
            )
        )

    return df


def create_cross_sensor_features(df):

    section("5. CROSS-SENSOR FEATURES")

    # Temperature-Humidity relationship
    df["T2M_RH2M_ratio"] = (
        df["T2M"]
        /
        df["RH2M"].replace(
            0,
            np.nan
        )
    )

    # Temperature-Pressure relationship
    df["T2M_PS_ratio"] = (
        df["T2M"]
        /
        df["PS"].replace(
            0,
            np.nan
        )
    )

    # Sensor normalized differences

    df["T2M_RH2M_difference"] = (
        (
            df["T2M"]
            - df["T2M"].mean()
        )
        -
        (
            df["RH2M"]
            - df["RH2M"].mean()
        )
    )

    df["T2M_PS_difference"] = (
        (
            df["T2M"]
            - df["T2M"].mean()
        )
        -
        (
            df["PS"]
            - df["PS"].mean()
        )
    )

    return df


def clean_features(df):

    section("6. FEATURE CLEANING")

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        )
        .columns
    )

    df[numeric_columns] = (
        df[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    before = len(df)

    df = df.dropna(
        subset=[
            "T2M_rolling_mean_6",
            "RH2M_rolling_mean_6",
            "PS_rolling_mean_6"
        ]
    )

    df = df.reset_index(
        drop=True
    )

    print(
        f"Rows before: {before:,}"
    )

    print(
        f"Rows after : {len(df):,}"
    )

    print(
        f"Rows removed: "
        f"{before - len(df):,}"
    )

    return df


def save_features(df):

    section("7. SAVING FEATURE DATA")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print(
        f"Shape: "
        f"{df.shape[0]:,} × "
        f"{df.shape[1]}"
    )


def main():

    print(
        """
╔══════════════════════════════════════════════════════════════╗
║                  ATMOS FEATURE ENGINE                       ║
║       Leakage-Safe Weather Feature Generation               ║
╚══════════════════════════════════════════════════════════════╝
"""
    )

    df = load_data()

    df = create_temporal_features(df)

    df = create_sensor_features(df)

    df = create_deviation_features(df)

    df = create_cross_sensor_features(df)

    df = clean_features(df)

    save_features(df)

    section("FEATURE ENGINE COMPLETE")

    print("STATUS: SUCCESS")


if __name__ == "__main__":
    main()