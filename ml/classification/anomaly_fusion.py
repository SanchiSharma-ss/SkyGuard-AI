import os
import json
import numpy as np
import pandas as pd


# ============================================================
# ATMOS — M8.1 CALIBRATED ANOMALY FUSION ENGINE
#
# Z-Score + IQR + Isolation Forest + Autoencoder
# Detector Voting + Calibrated Evidence + Severity
# Station-Safe Sensor Analysis
# ============================================================


BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)


BASELINE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "baseline",
    "baseline_predictions_test.csv"
)


AUTOENCODER_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "autoencoder",
    "autoencoder_predictions_test.csv"
)


FEATURE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ml_features.csv"
)


OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "fusion"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

# Minimum number of independent detectors required
# for a normal anomaly alert.
#
# 1 = sensitive
# 2 = balanced
# 3 = conservative
#
# We use 2 for the initial calibrated system.
MIN_DETECTOR_VOTES = 2


# Fusion score is used as supporting evidence,
# not as the only anomaly decision.
MIN_FUSION_SCORE = 0.30


# ============================================================
# HEADER
# ============================================================

print()

print(
    "╔══════════════════════════════════════════════════════════════╗"
)

print(
    "║             ATMOS CALIBRATED FUSION ENGINE                 ║"
)

print(
    "║      Multi-Detector Voting + Explainable Alerts            ║"
)

print(
    "╚══════════════════════════════════════════════════════════════╝"
)

print()


# ============================================================
# 1. LOAD BASELINE
# ============================================================

print("=" * 70)
print("1. LOADING BASELINE PREDICTIONS")
print("=" * 70)


baseline = pd.read_csv(
    BASELINE_FILE
)


baseline["timestamp"] = pd.to_datetime(
    baseline["timestamp"],
    errors="coerce"
)


print(
    f"Baseline records: "
    f"{len(baseline):,}"
)


# ============================================================
# 2. LOAD AUTOENCODER
# ============================================================

print()
print("=" * 70)
print("2. LOADING AUTOENCODER PREDICTIONS")
print("=" * 70)


autoencoder = pd.read_csv(
    AUTOENCODER_FILE
)


autoencoder["timestamp"] = pd.to_datetime(
    autoencoder["timestamp"],
    errors="coerce"
)


print(
    f"Autoencoder records: "
    f"{len(autoencoder):,}"
)


# ============================================================
# 3. LOAD FEATURE DATA
# ============================================================

print()
print("=" * 70)
print("3. LOADING ENGINEERED FEATURES")
print("=" * 70)


features = pd.read_csv(
    FEATURE_FILE
)


features["timestamp"] = pd.to_datetime(
    features["timestamp"],
    errors="coerce"
)


print(
    f"Feature records: "
    f"{len(features):,}"
)


# ============================================================
# 4. VERIFY REQUIRED COLUMNS
# ============================================================

print()
print("=" * 70)
print("4. VERIFYING MODEL OUTPUTS")
print("=" * 70)


required_baseline = [

    "timestamp",
    "station",

    "T2M",
    "RH2M",
    "PS",

    "actual_anomaly",

    "zscore_anomaly",
    "zscore_score",

    "iqr_anomaly",
    "iqr_score",

    "isolation_forest_anomaly",
    "isolation_forest_score",
]


required_autoencoder = [

    "timestamp",
    "station",

    "autoencoder_anomaly",
    "reconstruction_error",
]


required_features = [

    "timestamp",
    "station",

    "T2M_diff",
    "RH2M_diff",
    "PS_diff",

    "T2M_pct_change",
    "RH2M_pct_change",
    "PS_pct_change",

    "T2M_local_z",
    "RH2M_local_z",
    "PS_local_z",
]


for column in required_baseline:

    if column not in baseline.columns:

        raise ValueError(
            f"Missing baseline column: {column}"
        )


for column in required_autoencoder:

    if column not in autoencoder.columns:

        raise ValueError(
            f"Missing Autoencoder column: {column}"
        )


for column in required_features:

    if column not in features.columns:

        raise ValueError(
            f"Missing feature column: {column}"
        )


print(
    "All required columns verified."
)


# ============================================================
# 5. MERGE MODEL OUTPUTS
# ============================================================

print()
print("=" * 70)
print("5. MERGING MODEL OUTPUTS")
print("=" * 70)


df = baseline.merge(

    autoencoder[
        [
            "timestamp",
            "station",

            "autoencoder_anomaly",
            "reconstruction_error",
        ]
    ],

    on=[
        "timestamp",
        "station"
    ],

    how="inner"
)


# Add engineered features
df = df.merge(

    features[
        [
            "timestamp",
            "station",

            "T2M_diff",
            "RH2M_diff",
            "PS_diff",

            "T2M_pct_change",
            "RH2M_pct_change",
            "PS_pct_change",

            "T2M_local_z",
            "RH2M_local_z",
            "PS_local_z",
        ]
    ],

    on=[
        "timestamp",
        "station"
    ],

    how="left"
)


print(
    f"Fusion records: "
    f"{len(df):,}"
)


if len(df) == 0:

    raise ValueError(
        "No records matched across model outputs."
    )


# ============================================================
# 6. CLEAN NUMERIC VALUES
# ============================================================

print()
print("=" * 70)
print("6. CLEANING MODEL SCORES")
print("=" * 70)


numeric_columns = [

    "zscore_score",
    "iqr_score",
    "isolation_forest_score",
    "reconstruction_error",

    "T2M_diff",
    "RH2M_diff",
    "PS_diff",

    "T2M_pct_change",
    "RH2M_pct_change",
    "PS_pct_change",

    "T2M_local_z",
    "RH2M_local_z",
    "PS_local_z",
]


for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


df[numeric_columns] = (
    df[numeric_columns]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)


# ============================================================
# 7. DETECTOR VOTING
# ============================================================

print()
print("=" * 70)
print("7. CALCULATING DETECTOR VOTES")
print("=" * 70)


DETECTORS = [

    "zscore_anomaly",
    "iqr_anomaly",
    "isolation_forest_anomaly",
    "autoencoder_anomaly",

]


df["detector_votes"] = (
    df[DETECTORS]
    .astype(int)
    .sum(axis=1)
)


print(
    "Detector vote distribution:"
)

print(
    df[
        "detector_votes"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# 8. ROBUST SCORE NORMALIZATION
# ============================================================

print()
print("=" * 70)
print("8. CALIBRATING DETECTOR SCORES")
print("=" * 70)


def percentile_score(series):

    """
    Convert a raw anomaly score into a percentile-based
    anomaly strength.

    This avoids dependence on the absolute min/max
    values of the test batch.
    """

    values = (
        series
        .rank(
            pct=True,
            method="average"
        )
    )

    return values.fillna(0.0)


df["zscore_strength"] = (
    percentile_score(
        df["zscore_score"]
    )
)


df["iqr_strength"] = (
    percentile_score(
        df["iqr_score"]
    )
)


df["isolation_strength"] = (
    percentile_score(
        df["isolation_forest_score"]
    )
)


df["autoencoder_strength"] = (
    percentile_score(
        df["reconstruction_error"]
    )
)


print(
    "Detector scores converted to percentile strength."
)


# ============================================================
# 9. WEIGHTED FUSION SCORE
# ============================================================

print()
print("=" * 70)
print("9. CALCULATING WEIGHTED FUSION SCORE")
print("=" * 70)


# Autoencoder receives the largest weight because it
# models multivariate normal reconstruction.
#
# Isolation Forest provides a second multivariate signal.
#
# Z-score and IQR provide statistical evidence.

WEIGHT_Z = 0.20
WEIGHT_IQR = 0.10
WEIGHT_IF = 0.25
WEIGHT_AE = 0.45


df["fusion_score"] = (

    WEIGHT_Z
    * df["zscore_strength"]

    +

    WEIGHT_IQR
    * df["iqr_strength"]

    +

    WEIGHT_IF
    * df["isolation_strength"]

    +

    WEIGHT_AE
    * df["autoencoder_strength"]

)


# ============================================================
# 10. VOTE-BASED ANOMALY DECISION
# ============================================================

print()
print("=" * 70)
print("10. MAKING ANOMALY DECISION")
print("=" * 70)


# Important:
#
# We DO NOT require fusion_score >= 0.50 anymore.
#
# Detector agreement is the primary signal.
# Fusion score provides additional confidence.

df["anomaly_status"] = np.where(

    (
        (
            df["detector_votes"]
            >= MIN_DETECTOR_VOTES
        )

        &

        (
            df["fusion_score"]
            >= MIN_FUSION_SCORE
        )
    ),

    "ANOMALY",

    "NORMAL"
)


print(
    df[
        "anomaly_status"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# 11. DETECTOR AGREEMENT
# ============================================================

print()
print("=" * 70)
print("11. DETECTOR AGREEMENT")
print("=" * 70)


def agreement_label(votes):

    if votes == 4:
        return "VERY_STRONG"

    if votes == 3:
        return "STRONG"

    if votes == 2:
        return "MODERATE"

    if votes == 1:
        return "WEAK"

    return "NONE"


df["agreement_level"] = (
    df["detector_votes"]
    .apply(
        agreement_label
    )
)


# ============================================================
# 12. SEVERITY
# ============================================================

print()
print("=" * 70)
print("12. CALCULATING SEVERITY")
print("=" * 70)


def calculate_severity(row):

    if row["anomaly_status"] == "NORMAL":

        return "NORMAL"


    votes = int(
        row["detector_votes"]
    )

    score = float(
        row["fusion_score"]
    )


    # Four detectors agreeing
    if votes == 4:

        if score >= 0.75:
            return "CRITICAL"

        return "HIGH"


    # Three detectors agreeing
    if votes == 3:

        if score >= 0.75:
            return "HIGH"

        return "MEDIUM"


    # Two detectors agreeing
    if votes == 2:

        if score >= 0.75:
            return "HIGH"

        return "MEDIUM"


    return "LOW"


df["severity"] = df.apply(
    calculate_severity,
    axis=1
)


print(
    df[
        "severity"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# 13. CONFIDENCE
# ============================================================

print()
print("=" * 70)
print("13. CALCULATING CONFIDENCE")
print("=" * 70)


def calculate_confidence(row):

    votes = int(
        row["detector_votes"]
    )

    score = float(
        row["fusion_score"]
    )


    vote_confidence = {

        0: 0.05,
        1: 0.30,
        2: 0.60,
        3: 0.82,
        4: 0.95,

    }.get(
        votes,
        0.05
    )


    confidence = (

        0.65
        * vote_confidence

        +

        0.35
        * score

    )


    return round(
        np.clip(
            confidence,
            0,
            0.99
        )
        * 100,
        2
    )


df["confidence_percent"] = (
    df.apply(
        calculate_confidence,
        axis=1
    )
)


# ============================================================
# 14. STATION-SAFE SENSOR ANALYSIS
# ============================================================

print()
print("=" * 70)
print("14. STATION-SAFE SENSOR ANALYSIS")
print("=" * 70)


# IMPORTANT:
# Calculate temporal changes WITHIN each station.
# This prevents one station's reading from being compared
# with another station's reading.

df = df.sort_values(
    [
        "station",
        "timestamp"
    ]
).reset_index(
    drop=True
)


for sensor in [
    "T2M",
    "RH2M",
    "PS"
]:

    df[
        f"{sensor}_station_diff"
    ] = (

        df
        .groupby(
            "station"
        )[sensor]
        .diff()
        .abs()

    )


# ============================================================
# 15. IDENTIFY SUSPECTED SENSOR
# ============================================================

print()
print("=" * 70)
print("15. IDENTIFYING SUSPECTED SENSOR")
print("=" * 70)


def identify_sensor(row):

    # First use local-z magnitude when available.
    local_scores = {

        "T2M":
            abs(
                row["T2M_local_z"]
            ),

        "RH2M":
            abs(
                row["RH2M_local_z"]
            ),

        "PS":
            abs(
                row["PS_local_z"]
            ),

    }


    return max(
        local_scores,
        key=local_scores.get
    )


df["suspected_sensor"] = (
    df.apply(
        identify_sensor,
        axis=1
    )
)


# ============================================================
# 16. SENSOR EVIDENCE
# ============================================================

print()
print("=" * 70)
print("16. GENERATING SENSOR EVIDENCE")
print("=" * 70)


def sensor_evidence(row):

    sensor = row[
        "suspected_sensor"
    ]


    evidence = []


    if sensor == "T2M":

        if abs(
            row["T2M_local_z"]
        ) >= 3:

            evidence.append(
                "T2M strong local deviation"
            )

        if (
            row["T2M_diff"]
            != 0
        ):

            evidence.append(
                "T2M temporal change detected"
            )


    elif sensor == "RH2M":

        if abs(
            row["RH2M_local_z"]
        ) >= 3:

            evidence.append(
                "RH2M strong local deviation"
            )

        if (
            row["RH2M_diff"]
            != 0
        ):

            evidence.append(
                "RH2M temporal change detected"
            )


    elif sensor == "PS":

        if abs(
            row["PS_local_z"]
        ) >= 3:

            evidence.append(
                "PS strong local deviation"
            )

        if (
            row["PS_diff"]
            != 0
        ):

            evidence.append(
                "PS temporal change detected"
            )


    return evidence


df["sensor_evidence_list"] = (
    df.apply(
        sensor_evidence,
        axis=1
    )
)


# ============================================================
# 17. PRIMARY DETECTOR
# ============================================================

print()
print("=" * 70)
print("17. IDENTIFYING PRIMARY DETECTOR")
print("=" * 70)


def primary_detector(row):

    scores = {

        "Z-Score":
            row["zscore_strength"],

        "IQR":
            row["iqr_strength"],

        "Isolation Forest":
            row["isolation_strength"],

        "Autoencoder":
            row["autoencoder_strength"],

    }


    return max(
        scores,
        key=scores.get
    )


df["primary_detector"] = (
    df.apply(
        primary_detector,
        axis=1
    )
)


# ============================================================
# 18. EXPLAINABLE EVIDENCE
# ============================================================

print()
print("=" * 70)
print("18. GENERATING EXPLAINABLE ALERTS")
print("=" * 70)


def generate_evidence(row):

    evidence = []


    if row[
        "zscore_anomaly"
    ] == 1:

        evidence.append(
            "Statistical deviation"
        )


    if row[
        "iqr_anomaly"
    ] == 1:

        evidence.append(
            "IQR outlier"
        )


    if row[
        "isolation_forest_anomaly"
    ] == 1:

        evidence.append(
            "Multivariate isolation"
        )


    if row[
        "autoencoder_anomaly"
    ] == 1:

        evidence.append(
            "High reconstruction error"
        )


    if row[
        "detector_votes"
    ] >= 3:

        evidence.append(
            "Strong detector agreement"
        )


    sensor_evidence = row[
        "sensor_evidence_list"
    ]


    evidence.extend(
        sensor_evidence
    )


    if len(evidence) == 0:

        return (
            "No strong anomaly evidence"
        )


    # Remove duplicates while
    # preserving order.

    evidence = list(
        dict.fromkeys(
            evidence
        )
    )


    return " | ".join(
        evidence
    )


df["evidence"] = (
    df.apply(
        generate_evidence,
        axis=1
    )
)


# ============================================================
# 19. SAVE RESULTS
# ============================================================

print()
print("=" * 70)
print("19. SAVING FUSION RESULTS")
print("=" * 70)


output_columns = [

    "timestamp",
    "station",

    "T2M",
    "RH2M",
    "PS",

    "actual_anomaly",

    "zscore_anomaly",
    "zscore_score",

    "iqr_anomaly",
    "iqr_score",

    "isolation_forest_anomaly",
    "isolation_forest_score",

    "autoencoder_anomaly",
    "reconstruction_error",

    "detector_votes",
    "agreement_level",

    "fusion_score",

    "anomaly_status",

    "severity",

    "confidence_percent",

    "primary_detector",

    "suspected_sensor",

    "evidence",

]


fusion_output = df[
    output_columns
].copy()


fusion_output = fusion_output.sort_values(
    [
        "anomaly_status",
        "fusion_score"
    ],
    ascending=[
        True,
        False
    ]
)


output_file = os.path.join(
    OUTPUT_DIR,
    "fusion_predictions_test.csv"
)


fusion_output.to_csv(
    output_file,
    index=False
)


print(
    f"Saved: {output_file}"
)


# ============================================================
# 20. EVALUATE FUSION
# ============================================================

print()
print("=" * 70)
print("20. FUSION EVALUATION")
print("=" * 70)


y_true = (
    df[
        "actual_anomaly"
    ]
    .astype(int)
    .values
)


y_pred = (
    df[
        "anomaly_status"
    ]
    .eq("ANOMALY")
    .astype(int)
    .values
)


tp = int(
    (
        (y_true == 1)
        &
        (y_pred == 1)
    ).sum()
)


tn = int(
    (
        (y_true == 0)
        &
        (y_pred == 0)
    ).sum()
)


fp = int(
    (
        (y_true == 0)
        &
        (y_pred == 1)
    ).sum()
)


fn = int(
    (
        (y_true == 1)
        &
        (y_pred == 0)
    ).sum()
)


precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else 0
)


recall = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0
)


f1 = (

    2
    * precision
    * recall
    /
    (precision + recall)

    if (precision + recall) > 0

    else 0
)


fpr = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else 0
)


print()
print(
    f"Precision          : "
    f"{precision:.4f}"
)

print(
    f"Recall             : "
    f"{recall:.4f}"
)

print(
    f"F1 Score           : "
    f"{f1:.4f}"
)

print(
    f"False Positive Rate: "
    f"{fpr:.4f}"
)

print(
    f"True Positives     : "
    f"{tp:,}"
)

print(
    f"False Positives    : "
    f"{fp:,}"
)

print(
    f"True Negatives     : "
    f"{tn:,}"
)

print(
    f"False Negatives    : "
    f"{fn:,}"
)


# ============================================================
# 21. SUMMARY
# ============================================================

print()
print("=" * 70)
print("21. FUSION SUMMARY")
print("=" * 70)


predicted_anomalies = int(
    (
        df[
            "anomaly_status"
        ]
        == "ANOMALY"
    ).sum()
)


summary = {

    "total_records":
        int(len(df)),

    "actual_anomalies":
        int(
            df[
                "actual_anomaly"
            ].sum()
        ),

    "predicted_anomalies":
        predicted_anomalies,

    "precision":
        float(precision),

    "recall":
        float(recall),

    "f1":
        float(f1),

    "false_positive_rate":
        float(fpr),

    "true_positives":
        tp,

    "false_positives":
        fp,

    "true_negatives":
        tn,

    "false_negatives":
        fn,

    "average_fusion_score":
        float(
            df[
                "fusion_score"
            ].mean()
        ),

    "average_confidence":
        float(
            df[
                "confidence_percent"
            ].mean()
        ),

    "detector_vote_distribution":
        {
            str(k): int(v)

            for k, v in
            df[
                "detector_votes"
            ]
            .value_counts()
            .sort_index()
            .items()
        },

    "severity_distribution":
        {
            str(k): int(v)

            for k, v in
            df[
                "severity"
            ]
            .value_counts()
            .items()
        },

}


summary_file = os.path.join(
    OUTPUT_DIR,
    "fusion_summary.json"
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


print(
    f"Saved: {summary_file}"
)


# ============================================================
# 22. TOP ALERTS
# ============================================================

print()
print("=" * 70)
print("22. TOP ATMOS ALERTS")
print("=" * 70)


alerts = fusion_output[
    fusion_output[
        "anomaly_status"
    ]
    == "ANOMALY"
].sort_values(
    "fusion_score",
    ascending=False
)


if len(alerts) > 0:

    alert_columns = [

        "timestamp",
        "station",
        "suspected_sensor",
        "severity",
        "confidence_percent",
        "fusion_score",
        "detector_votes",
        "primary_detector",
        "evidence",

    ]


    print(
        alerts[
            alert_columns
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

else:

    print(
        "No anomaly alerts generated."
    )


# ============================================================
# 23. FINAL STATUS
# ============================================================

print()
print("=" * 70)
print("23. M8.1 FINAL STATUS")
print("=" * 70)


print(
    f"Total records       : "
    f"{len(df):,}"
)

print(
    f"Actual anomalies    : "
    f"{int(df['actual_anomaly'].sum()):,}"
)

print(
    f"Predicted anomalies : "
    f"{predicted_anomalies:,}"
)

print(
    f"Precision           : "
    f"{precision:.4f}"
)

print(
    f"Recall              : "
    f"{recall:.4f}"
)

print(
    f"F1                  : "
    f"{f1:.4f}"
)

print(
    f"False Positive Rate : "
    f"{fpr:.4f}"
)


print()
print("Output files:")

print(
    "  ✓ data/processed/fusion/"
    "fusion_predictions_test.csv"
)

print(
    "  ✓ data/processed/fusion/"
    "fusion_summary.json"
)


print()

print(
    "╔══════════════════════════════════════════════════════════════╗"
)

print(
    "║            M8.1 CALIBRATED FUSION COMPLETE                 ║"
)

print(
    "║                    STATUS: SUCCESS                         ║"
)

print(
    "╚══════════════════════════════════════════════════════════════╝"
)

print()