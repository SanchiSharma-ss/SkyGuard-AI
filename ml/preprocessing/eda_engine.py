from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# ATMOS - EDA ENGINE
# ============================================================

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
    / "eda"
)


def section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def load_data():

    section("1. LOADING CLEAN DATA")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    return df


def station_analysis(df):

    section("2. STATION ANALYSIS")

    if "station" not in df.columns:
        print("Station column not found.")
        return None

    result = (
        df.groupby("station")
        .agg(
            records=("station", "size"),
            anomalies=("any_anomaly", "sum")
        )
        .sort_values("records", ascending=False)
    )

    result["anomaly_rate"] = (
        result["anomalies"] / result["records"] * 100
    )

    print(result.to_string())

    return result


def anomaly_analysis(df):

    section("3. ANOMALY ANALYSIS")

    anomaly_columns = [
        "temperature_anomaly",
        "humidity_anomaly",
        "pressure_anomaly",
        "any_anomaly"
    ]

    results = {}

    for column in anomaly_columns:

        if column in df.columns:

            count = int(df[column].sum())

            percentage = (
                count / len(df) * 100
            )

            results[column] = {
                "count": count,
                "percentage": percentage
            }

            print(
                f"{column:30s}"
                f" {count:6,d}"
                f" ({percentage:.2f}%)"
            )

    return results


def fault_analysis(df):

    section("4. FAULT TYPE ANALYSIS")

    fault_columns = [
        "temperature_fault_type",
        "humidity_fault_type",
        "pressure_fault_type"
    ]

    results = {}

    for column in fault_columns:

        if column not in df.columns:
            continue

        print(f"\n{column}")

        counts = (
            df[column]
            .fillna("NONE")
            .astype(str)
            .value_counts()
        )

        print(counts.to_string())

        results[column] = {
            str(k): int(v)
            for k, v in counts.items()
        }

    return results


def split_analysis(df):

    section("5. DATA SPLIT ANALYSIS")

    if "split" not in df.columns:
        print("Split column not found.")
        return None

    result = (
        df["split"]
        .value_counts()
        .to_dict()
    )

    for key, value in result.items():

        percentage = (
            value / len(df) * 100
        )

        print(
            f"{key:15s}"
            f"{value:8,d}"
            f" ({percentage:.2f}%)"
        )

    return {
        str(k): int(v)
        for k, v in result.items()
    }


def sensor_statistics(df):

    section("6. SENSOR STATISTICS")

    sensors = [
        "T2M",
        "RH2M",
        "PS"
    ]

    available = [
        column
        for column in sensors
        if column in df.columns
    ]

    if not available:
        print("No core sensors found.")
        return None

    stats = (
        df[available]
        .describe()
        .transpose()
    )

    print(stats.to_string())

    return stats


def correlation_analysis(df):

    section("7. SENSOR CORRELATION")

    sensors = [
        "T2M",
        "RH2M",
        "PS"
    ]

    available = [
        column
        for column in sensors
        if column in df.columns
    ]

    correlation = (
        df[available]
        .corr()
    )

    print(correlation.to_string())

    return correlation


def anomaly_by_station(df):

    section("8. ANOMALIES BY STATION")

    if (
        "station" not in df.columns
        or "any_anomaly" not in df.columns
    ):
        return None

    result = (
        df.groupby("station")
        .agg(
            records=("station", "size"),
            anomalies=("any_anomaly", "sum")
        )
    )

    result["anomaly_rate"] = (
        result["anomalies"]
        / result["records"]
        * 100
    )

    result = result.sort_values(
        "anomalies",
        ascending=False
    )

    print(result.to_string())

    return result


def extreme_value_check(df):

    section("9. EXTREME VALUE CHECK")

    sensors = [
        "T2M",
        "RH2M",
        "PS"
    ]

    results = {}

    for sensor in sensors:

        if sensor not in df.columns:
            continue

        q1 = df[sensor].quantile(0.25)
        q3 = df[sensor].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = int(
            (
                (df[sensor] < lower)
                |
                (df[sensor] > upper)
            ).sum()
        )

        results[sensor] = {
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "potential_outliers": count
        }

        print(
            f"{sensor}: "
            f"{count:,} potential IQR outliers"
        )

    return results


def save_outputs(
    station_data,
    anomaly_data,
    fault_data,
    split_data,
    extreme_data
):

    section("10. SAVING EDA RESULTS")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if station_data is not None:
        station_data.to_csv(
            OUTPUT_DIR / "station_summary.csv"
        )

    if anomaly_data is not None:
        with open(
            OUTPUT_DIR / "anomaly_summary.json",
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                anomaly_data,
                file,
                indent=4
            )

    if fault_data is not None:
        with open(
            OUTPUT_DIR / "fault_summary.json",
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                fault_data,
                file,
                indent=4
            )

    if split_data is not None:
        with open(
            OUTPUT_DIR / "split_summary.json",
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                split_data,
                file,
                indent=4
            )

    if extreme_data is not None:
        with open(
            OUTPUT_DIR / "extreme_value_summary.json",
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                extreme_data,
                file,
                indent=4
            )

    print(f"EDA results saved to:")
    print(OUTPUT_DIR)


def main():

    print(
        """
╔══════════════════════════════════════════════════════════════╗
║                     ATMOS EDA ENGINE                        ║
║        Weather Data Intelligence & Analysis                 ║
╚══════════════════════════════════════════════════════════════╝
"""
    )

    df = load_data()

    station_data = station_analysis(df)

    anomaly_data = anomaly_analysis(df)

    fault_data = fault_analysis(df)

    split_data = split_analysis(df)

    sensor_statistics(df)

    correlation_analysis(df)

    anomaly_by_station(df)

    extreme_data = extreme_value_check(df)

    save_outputs(
        station_data,
        anomaly_data,
        fault_data,
        split_data,
        extreme_data
    )

    section("EDA COMPLETE")

    print("STATUS: SUCCESS")


if __name__ == "__main__":
    main()