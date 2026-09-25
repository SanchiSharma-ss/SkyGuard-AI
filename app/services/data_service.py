import os
import pandas as pd
import numpy as np


# ============================================================
# ATMOS DATA SERVICE
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)


FEATURE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ml_features.csv"
)

HEALTH_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "health",
    "sensor_health_scores.csv"
)

STATION_HEALTH_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "health",
    "station_health_scores.csv"
)

CLASSIFICATION_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "classification",
    "fault_predictions_test.csv"
)


# ============================================================
# HELPERS
# ============================================================

def clean_station(value):

    if pd.isna(value):
        return ""

    value = str(value).strip()

    # Fix tuple-like station values such as:
    # ('silchar',)
    if value.startswith("(") and value.endswith(",)"):
        value = value[1:-2]

    value = value.strip("'\"")

    return value


def clean_value(value):

    if pd.isna(value):
        return None

    if isinstance(value, np.generic):
        value = value.item()

    return value


def records_to_json(df):

    records = df.to_dict(
        orient="records"
    )

    cleaned = []

    for record in records:

        cleaned_record = {
            str(k): clean_value(v)
            for k, v in record.items()
        }

        cleaned.append(
            cleaned_record
        )

    return cleaned


# ============================================================
# LOAD DATA
# ============================================================

def load_features():

    df = pd.read_csv(
        FEATURE_FILE
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df["station"] = (
        df["station"]
        .apply(clean_station)
    )

    return df


def load_health():

    df = pd.read_csv(
        HEALTH_FILE
    )

    df["station"] = (
        df["station"]
        .apply(clean_station)
    )

    return df


def load_station_health():

    df = pd.read_csv(
        STATION_HEALTH_FILE
    )

    df["station"] = (
        df["station"]
        .apply(clean_station)
    )

    return df


def load_classification():

    df = pd.read_csv(
        CLASSIFICATION_FILE
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df["station"] = (
        df["station"]
        .apply(clean_station)
    )

    return df


# ============================================================
# STATIONS
# ============================================================

def get_stations():

    df = load_features()

    stations = (
        df["station"]
        .dropna()
        .unique()
        .tolist()
    )

    stations = sorted(
        stations
    )

    return stations


def get_station(station):

    df = load_features()

    station = clean_station(
        station
    )

    result = df[
        df["station"].str.lower()
        == station.lower()
    ]

    return result


# ============================================================
# LATEST OBSERVATION
# ============================================================

def get_latest_station_data(
    station
):

    df = get_station(
        station
    )

    if df.empty:
        return None

    df = df.sort_values(
        "timestamp"
    )

    row = df.iloc[-1]

    return {

        "timestamp":
            clean_value(
                row["timestamp"]
            ),

        "station":
            clean_station(
                row["station"]
            ),

        "T2M":
            clean_value(
                row["T2M"]
            ),

        "RH2M":
            clean_value(
                row["RH2M"]
            ),

        "PS":
            clean_value(
                row["PS"]
            ),

    }


# ============================================================
# SENSOR DATA
# ============================================================

def get_station_sensors(
    station
):

    health = load_health()

    station = clean_station(
        station
    )

    result = health[
        health["station"].str.lower()
        == station.lower()
    ]

    if result.empty:
        return []

    return records_to_json(
        result
    )


# ============================================================
# SENSOR HEALTH
# ============================================================

def get_sensor_health(
    station=None,
    sensor=None
):

    df = load_health()

    if station:

        station = clean_station(
            station
        )

        df = df[
            df["station"].str.lower()
            == station.lower()
        ]

    if sensor:

        df = df[
            df["sensor"].str.upper()
            == sensor.upper()
        ]

    return records_to_json(
        df
    )


# ============================================================
# STATION HEALTH
# ============================================================

def get_station_health(
    station=None
):

    df = load_station_health()

    if station:

        station = clean_station(
            station
        )

        df = df[
            df["station"].str.lower()
            == station.lower()
        ]

    return records_to_json(
        df
    )


# ============================================================
# ANOMALIES
# ============================================================

def get_anomalies(
    station=None,
    sensor=None,
    limit=100
):

    classification = load_classification()

    if station:

        station = clean_station(
            station
        )

        classification = classification[
            classification["station"].str.lower()
            == station.lower()
        ]

    anomaly_rows = []

    sensor_configs = {

        "T2M": {
            "fault":
                "T2M_predicted_fault",
            "confidence":
                "T2M_fault_confidence",
        },

        "RH2M": {
            "fault":
                "RH2M_predicted_fault",
            "confidence":
                "RH2M_fault_confidence",
        },

        "PS": {
            "fault":
                "PS_predicted_fault",
            "confidence":
                "PS_fault_confidence",
        },

    }

    requested_sensors = (
        [sensor.upper()]
        if sensor
        else list(
            sensor_configs.keys()
        )
    )

    for requested_sensor in requested_sensors:

        if requested_sensor not in sensor_configs:
            continue

        config = sensor_configs[
            requested_sensor
        ]

        fault_col = config["fault"]
        confidence_col = config["confidence"]

        if fault_col not in classification.columns:
            continue

        sensor_rows = classification[
            classification[fault_col]
            .astype(str)
            .str.lower()
            != "normal"
        ].copy()

        for _, row in sensor_rows.iterrows():

            anomaly_rows.append({

                "timestamp":
                    clean_value(
                        row["timestamp"]
                    ),

                "station":
                    clean_station(
                        row["station"]
                    ),

                "sensor":
                    requested_sensor,

                "fault":
                    clean_value(
                        row[fault_col]
                    ),

                "confidence":
                    clean_value(
                        row[confidence_col]
                    ),

            })

    result = pd.DataFrame(
        anomaly_rows
    )

    if result.empty:
        return []

    result = result.sort_values(
        "timestamp",
        ascending=False
    )

    result = result.head(
        int(limit)
    )

    return records_to_json(
        result
    )


# ============================================================
# DASHBOARD STATISTICS
# ============================================================

def get_statistics():

    features = load_features()

    health = load_health()

    stations = (
        features["station"]
        .nunique()
    )

    sensors = (
        stations * 3
    )

    active_anomalies = 0

    classification = load_classification()

    for sensor in [
        "T2M",
        "RH2M",
        "PS"
    ]:

        column = (
            f"{sensor}_predicted_fault"
        )

        if column in classification.columns:

            active_anomalies += int(
                (
                    classification[column]
                    .astype(str)
                    .str.lower()
                    != "normal"
                ).sum()
            )

    return {

        "stations":
            int(stations),

        "sensors":
            int(sensors),

        "active_anomalies":
            int(active_anomalies),

        "average_sensor_health":
            round(
                float(
                    health[
                        "health_score"
                    ].mean()
                ),
                2
            ),

        "excellent_sensors":
            int(
                (
                    health["status"]
                    == "EXCELLENT"
                ).sum()
            ),

        "healthy_sensors":
            int(
                (
                    health["status"]
                    == "HEALTHY"
                ).sum()
            ),

        "warning_sensors":
            int(
                (
                    health["status"]
                    == "WARNING"
                ).sum()
            ),

        "degraded_sensors":
            int(
                (
                    health["status"]
                    == "DEGRADED"
                ).sum()
            ),

        "critical_sensors":
            int(
                (
                    health["status"]
                    == "CRITICAL"
                ).sum()
            ),

    }