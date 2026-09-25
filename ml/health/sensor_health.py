import os
import json
import numpy as np
import pandas as pd


# ============================================================
# ATMOS — M10 SENSOR HEALTH ENGINE
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

FUSION_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "baseline",
    "fusion_predictions_test.csv"
)

CLASSIFICATION_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "classification",
    "fault_predictions_test.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "health"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print()

print(
    "╔══════════════════════════════════════════════════════════════╗"
)

print(
    "║                 ATMOS SENSOR HEALTH ENGINE                ║"
)

print(
    "║                    M10 HEALTH SCORING                     ║"
)

print(
    "╚══════════════════════════════════════════════════════════════╝"
)

print()


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("1. LOADING DATA")
print("=" * 70)


features = pd.read_csv(
    FEATURE_FILE
)

features["timestamp"] = pd.to_datetime(
    features["timestamp"],
    errors="coerce"
)


classification = pd.read_csv(
    CLASSIFICATION_FILE
)

classification["timestamp"] = pd.to_datetime(
    classification["timestamp"],
    errors="coerce"
)


print(
    f"Feature records       : {len(features):,}"
)

print(
    f"Classification records: {len(classification):,}"
)


# ============================================================
# 2. PREPARE SENSOR INFORMATION
# ============================================================

print()
print("=" * 70)
print("2. PREPARING SENSOR HEALTH FEATURES")
print("=" * 70)


SENSORS = {

    "T2M": {
        "actual": "T2M_actual_fault",
        "predicted": "T2M_predicted_fault",
        "confidence": "T2M_fault_confidence",
        "local_z": "T2M_local_z",
        "diff": "T2M_diff",
        "rolling_std": "T2M_rolling_std_6",
    },

    "RH2M": {
        "actual": "RH2M_actual_fault",
        "predicted": "RH2M_predicted_fault",
        "confidence": "RH2M_fault_confidence",
        "local_z": "RH2M_local_z",
        "diff": "RH2M_diff",
        "rolling_std": "RH2M_rolling_std_6",
    },

    "PS": {
        "actual": "PS_actual_fault",
        "predicted": "PS_predicted_fault",
        "confidence": "PS_fault_confidence",
        "local_z": "PS_local_z",
        "diff": "PS_diff",
        "rolling_std": "PS_rolling_std_6",
    },

}


# ============================================================
# 3. MERGE CLASSIFICATION WITH FEATURES
# ============================================================

print()
print("=" * 70)
print("3. MERGING CLASSIFICATION RESULTS")
print("=" * 70)


merge_columns = [
    "timestamp",
    "station",
]


available_classification = [
    column
    for column in classification.columns
    if column not in merge_columns
]


health_df = features.merge(
    classification[
        merge_columns + available_classification
    ],
    on=merge_columns,
    how="left"
)


print(
    f"Merged records: "
    f"{len(health_df):,}"
)


# ============================================================
# 4. HEALTH CALCULATION FUNCTIONS
# ============================================================

def clip_score(value):
    return float(
        np.clip(
            value,
            0,
            100
        )
    )


def calculate_health(
    anomaly_rate,
    severity,
    fault_confidence,
    fault_present,
    local_deviation,
    volatility,
    recovery
):
    """
    Calculate sensor health from 0–100.

    Higher anomaly frequency,
    severity,
    fault confidence,
    local deviation,
    and volatility reduce health.

    Recovery improves health.
    """

    anomaly_penalty = min(
        anomaly_rate * 35,
        35
    )

    severity_penalty = min(
        severity * 25,
        25
    )

    fault_penalty = min(
        fault_confidence *
        fault_present *
        0.20,
        20
    )

    deviation_penalty = min(
        local_deviation * 5,
        10
    )

    volatility_penalty = min(
        volatility * 2,
        5
    )

    recovery_bonus = min(
        recovery * 5,
        5
    )

    score = (
        100
        - anomaly_penalty
        - severity_penalty
        - fault_penalty
        - deviation_penalty
        - volatility_penalty
        + recovery_bonus
    )

    return clip_score(score)


def health_status(score):

    if score >= 90:
        return "EXCELLENT"

    if score >= 75:
        return "HEALTHY"

    if score >= 60:
        return "WARNING"

    if score >= 40:
        return "DEGRADED"

    return "CRITICAL"


def severity_value(fault):

    fault = str(fault).lower()

    if fault in ["normal", "nan"]:
        return 0.0

    if fault == "drift":
        return 0.60

    if fault == "stuck":
        return 0.70

    if fault == "drop":
        return 0.80

    if fault == "spike":
        return 0.90

    return 0.50


# ============================================================
# 5. CALCULATE HEALTH
# ============================================================

print()
print("=" * 70)
print("4. CALCULATING SENSOR HEALTH")
print("=" * 70)


health_records = []


group_columns = [
    "station"
]


for station, station_df in health_df.groupby(
    group_columns
):

    station_df = station_df.sort_values(
        "timestamp"
    )


    for sensor, config in SENSORS.items():

        predicted_col = config["predicted"]
        confidence_col = config["confidence"]
        z_col = config["local_z"]
        diff_col = config["diff"]
        std_col = config["rolling_std"]


        if predicted_col not in station_df.columns:

            continue


        faults = (
            station_df[
                predicted_col
            ]
            .fillna("normal")
            .astype(str)
            .str.lower()
        )


        confidence = pd.to_numeric(
            station_df[
                confidence_col
            ],
            errors="coerce"
        ).fillna(0)


        local_z = pd.to_numeric(
            station_df[
                z_col
            ],
            errors="coerce"
        ).fillna(0)


        sensor_diff = pd.to_numeric(
            station_df[
                diff_col
            ],
            errors="coerce"
        ).fillna(0)


        rolling_std = pd.to_numeric(
            station_df[
                std_col
            ],
            errors="coerce"
        ).fillna(0)


        # ----------------------------------------------------
        # ANOMALY RATE
        # ----------------------------------------------------

        fault_mask = (
            faults != "normal"
        )

        anomaly_rate = (
            fault_mask.mean()
        )


        # ----------------------------------------------------
        # AVERAGE SEVERITY
        # ----------------------------------------------------

        severity_series = faults.map(
            severity_value
        )

        average_severity = (
            severity_series.mean()
        )


        # ----------------------------------------------------
        # FAULT CONFIDENCE
        # ----------------------------------------------------

        fault_confidence = (
            confidence[
                fault_mask
            ].mean()
            if fault_mask.any()
            else 0
        )


        # ----------------------------------------------------
        # LOCAL DEVIATION
        # ----------------------------------------------------

        abs_z = np.abs(
            local_z
        )

        local_deviation = np.clip(
            abs_z.mean(),
            0,
            2
        )


        # ----------------------------------------------------
        # VOLATILITY
        # ----------------------------------------------------

        volatility = np.clip(
            rolling_std.mean(),
            0,
            5
        )


        # ----------------------------------------------------
        # RECOVERY
        # ----------------------------------------------------

        if len(faults) > 1:

            normal_after_fault = (
                fault_mask.shift(1).fillna(False)
                & (~fault_mask)
            )

            recovery = (
                normal_after_fault.mean()
            )

        else:

            recovery = 0


        # ----------------------------------------------------
        # HEALTH SCORE
        # ----------------------------------------------------

        score = calculate_health(

            anomaly_rate=anomaly_rate,

            severity=average_severity,

            fault_confidence=fault_confidence,

            fault_present=float(
                fault_mask.mean()
            ),

            local_deviation=local_deviation,

            volatility=volatility,

            recovery=recovery

        )


        status = health_status(
            score
        )


        # ----------------------------------------------------
        # DOMINANT FAULT
        # ----------------------------------------------------

        fault_counts = (
            faults[
                fault_mask
            ]
            .value_counts()
        )


        if len(fault_counts) > 0:

            dominant_fault = (
                fault_counts
                .idxmax()
            )

        else:

            dominant_fault = "normal"


        # ----------------------------------------------------
        # RECORD
        # ----------------------------------------------------

        health_records.append({

            "station":
                station,

            "sensor":
                sensor,

            "health_score":
                round(score, 2),

            "status":
                status,

            "anomaly_rate":
                round(
                    anomaly_rate * 100,
                    3
                ),

            "average_severity":
                round(
                    average_severity,
                    4
                ),

            "fault_confidence":
                round(
                    float(
                        fault_confidence
                    ),
                    2
                ),

            "local_deviation":
                round(
                    float(
                        local_deviation
                    ),
                    4
                ),

            "volatility":
                round(
                    float(
                        volatility
                    ),
                    4
                ),

            "recovery_rate":
                round(
                    float(
                        recovery * 100
                    ),
                    3
                ),

            "dominant_fault":
                dominant_fault,

            "records":
                len(station_df),

        })


# ============================================================
# 6. CREATE HEALTH DATAFRAME
# ============================================================

print()
print("=" * 70)
print("5. CREATING HEALTH REPORT")
print("=" * 70)


health_report = pd.DataFrame(
    health_records
)


health_report = health_report.sort_values(
    [
        "health_score",
        "station",
        "sensor"
    ]
)


print(
    f"Health records generated: "
    f"{len(health_report):,}"
)


# ============================================================
# 7. SAVE SENSOR HEALTH
# ============================================================

health_file = os.path.join(
    OUTPUT_DIR,
    "sensor_health_scores.csv"
)


health_report.to_csv(
    health_file,
    index=False
)


print(
    f"Saved: {health_file}"
)


# ============================================================
# 8. STATION HEALTH
# ============================================================

print()
print("=" * 70)
print("6. CALCULATING STATION HEALTH")
print("=" * 70)


station_health = (
    health_report
    .groupby("station")
    .agg(

        station_health_score=(
            "health_score",
            "mean"
        ),

        minimum_sensor_health=(
            "health_score",
            "min"
        ),

        average_anomaly_rate=(
            "anomaly_rate",
            "mean"
        ),

        sensor_count=(
            "sensor",
            "count"
        ),

    )
    .reset_index()
)


station_health[
    "station_health_score"
] = station_health[
    "station_health_score"
].round(2)


station_health[
    "minimum_sensor_health"
] = station_health[
    "minimum_sensor_health"
].round(2)


station_health[
    "status"
] = station_health[
    "station_health_score"
].apply(
    health_status
)


station_health = station_health.sort_values(
    "station_health_score"
)


station_file = os.path.join(
    OUTPUT_DIR,
    "station_health_scores.csv"
)


station_health.to_csv(
    station_file,
    index=False
)


print(
    f"Saved: {station_file}"
)


# ============================================================
# 9. HEALTH SUMMARY
# ============================================================

print()
print("=" * 70)
print("7. HEALTH SUMMARY")
print("=" * 70)


print()
print(
    "Sensor health distribution:"
)

print(
    health_report[
        "status"
    ]
    .value_counts()
    .to_string()
)


print()
print(
    "Lowest-health sensors:"
)

print(
    health_report[
        [
            "station",
            "sensor",
            "health_score",
            "status",
            "dominant_fault",
            "anomaly_rate"
        ]
    ]
    .head(15)
    .to_string(
        index=False
    )
)


print()
print(
    "Station health:"
)

print(
    station_health[
        [
            "station",
            "station_health_score",
            "minimum_sensor_health",
            "status"
        ]
    ]
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# 10. SAVE JSON SUMMARY
# ============================================================

summary = {

    "total_sensor_health_records":
        int(
            len(
                health_report
            )
        ),

    "total_stations":
        int(
            health_report[
                "station"
            ].nunique()
        ),

    "average_sensor_health":
        round(
            float(
                health_report[
                    "health_score"
                ].mean()
            ),
            2
        ),

    "minimum_sensor_health":
        round(
            float(
                health_report[
                    "health_score"
                ].min()
            ),
            2
        ),

    "maximum_sensor_health":
        round(
            float(
                health_report[
                    "health_score"
                ].max()
            ),
            2
        ),

    "status_distribution":
        {
            str(k): int(v)

            for k, v in
            health_report[
                "status"
            ]
            .value_counts()
            .items()
        }

}


summary_file = os.path.join(
    OUTPUT_DIR,
    "health_summary.json"
)


with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


print()
print(
    f"Saved: {summary_file}"
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)

print(
    "M10 SENSOR HEALTH ENGINE COMPLETE"
)

print("=" * 70)

print()

print(
    "Generated:"
)

print(
    "  ✓ sensor_health_scores.csv"
)

print(
    "  ✓ station_health_scores.csv"
)

print(
    "  ✓ health_summary.json"
)

print()

print(
    "ATMOS health scoring is ready for API integration."
)

print()