import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# ATMOS — M6 BASELINE ANOMALY DETECTION
# Z-Score + IQR + Isolation Forest
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ml_features.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "baseline"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


print()
print("╔══════════════════════════════════════════════════════════════╗")
print("║                 ATMOS BASELINE ML ENGINE                    ║")
print("║       Z-Score + IQR + Isolation Forest                     ║")
print("╚══════════════════════════════════════════════════════════════╝")
print()


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("1. LOADING FEATURE DATA")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

print(f"Loaded {len(df):,} records")
print(f"Features available: {df.shape[1]}")


# ============================================================
# 2. DEFINE GROUND TRUTH
# ============================================================

print()
print("=" * 70)
print("2. DEFINING GROUND-TRUTH TARGET")
print("=" * 70)

TARGET = "any_anomaly"

if TARGET not in df.columns:
    raise ValueError(
        f"Required target column '{TARGET}' not found."
    )

df[TARGET] = (
    pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )
    .fillna(0)
    .astype(int)
)

print(f"Target: {TARGET}")
print(
    f"Normal records : "
    f"{(df[TARGET] == 0).sum():,}"
)
print(
    f"Anomaly records: "
    f"{(df[TARGET] == 1).sum():,}"
)


# ============================================================
# 3. SELECT LEAKAGE-SAFE FEATURES
# ============================================================

print()
print("=" * 70)
print("3. SELECTING LEAKAGE-SAFE FEATURES")
print("=" * 70)


# ------------------------------------------------------------
# RAW SENSOR FEATURES
# ------------------------------------------------------------

RAW_FEATURES = [
    "T2M",
    "RH2M",
    "PS",
]


# ------------------------------------------------------------
# ENGINEERED FEATURES
# These were generated independently in M5.
# ------------------------------------------------------------

ENGINEERED_FEATURES = [

    # Rate of change
    "T2M_diff",
    "RH2M_diff",
    "PS_diff",

    "T2M_pct_change",
    "RH2M_pct_change",
    "PS_pct_change",

    # Rolling statistics
    "T2M_rolling_mean_6",
    "RH2M_rolling_mean_6",
    "PS_rolling_mean_6",

    "T2M_rolling_std_6",
    "RH2M_rolling_std_6",
    "PS_rolling_std_6",

    # Local deviation
    "T2M_local_z",
    "RH2M_local_z",
    "PS_local_z",

    # Cross-sensor relationships
    "T2M_RH2M_ratio",
    "T2M_PS_ratio",

    "T2M_RH2M_difference",
    "T2M_PS_difference",
]


FEATURES = (
    RAW_FEATURES
    + ENGINEERED_FEATURES
)


missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:

    print()
    print("ERROR: Missing features:")
    for feature in missing_features:
        print(f"  ✗ {feature}")

    raise ValueError(
        f"Missing required features: {missing_features}"
    )


print(
    f"Selected features: "
    f"{len(FEATURES)}"
)

for feature in FEATURES:
    print(f"  ✓ {feature}")


# ============================================================
# 4. DATA SPLIT
# ============================================================

print()
print("=" * 70)
print("4. DATA SPLIT")
print("=" * 70)

if "split" not in df.columns:
    raise ValueError(
        "The 'split' column is required."
    )

train_df = df[
    df["split"] == "train"
].copy()

val_df = df[
    df["split"] == "validation"
].copy()

test_df = df[
    df["split"] == "test"
].copy()


print(
    f"Training   : "
    f"{len(train_df):,}"
)

print(
    f"Validation : "
    f"{len(val_df):,}"
)

print(
    f"Testing    : "
    f"{len(test_df):,}"
)


# ============================================================
# 5. PREPARE NUMERIC FEATURES
# ============================================================

print()
print("=" * 70)
print("5. PREPARING NUMERIC FEATURES")
print("=" * 70)


X_train = train_df[
    FEATURES
].replace(
    [np.inf, -np.inf],
    np.nan
)

X_val = val_df[
    FEATURES
].replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = test_df[
    FEATURES
].replace(
    [np.inf, -np.inf],
    np.nan
)


# Use TRAINING medians only
train_medians = X_train.median()

X_train = X_train.fillna(
    train_medians
)

X_val = X_val.fillna(
    train_medians
)

X_test = X_test.fillna(
    train_medians
)


print(
    "Missing values handled using "
    "training-set medians."
)


# ============================================================
# 6. NORMAL TRAINING DATA
# ============================================================

print()
print("=" * 70)
print("6. PREPARING NORMAL TRAINING DATA")
print("=" * 70)


normal_train_mask = (
    train_df[TARGET] == 0
)

X_train_normal = X_train.loc[
    normal_train_mask.values
].copy()


print(
    f"Normal training records: "
    f"{len(X_train_normal):,}"
)

print(
    f"Anomalous training records excluded: "
    f"{len(X_train) - len(X_train_normal):,}"
)


# ============================================================
# 7. Z-SCORE DETECTOR
# ============================================================

print()
print("=" * 70)
print("7. Z-SCORE ANOMALY DETECTOR")
print("=" * 70)


z_mean = X_train_normal.mean()

z_std = X_train_normal.std()

# Prevent division by zero
z_std = z_std.replace(
    0,
    1e-8
)


def calculate_zscore_anomaly(X):

    z_scores = np.abs(
        (X - z_mean) / z_std
    )

    # Maximum deviation across all features
    max_z = z_scores.max(
        axis=1
    )

    # Standard statistical threshold
    prediction = (
        max_z >= 3.0
    ).astype(int)

    return prediction, max_z


z_train_pred, z_train_score = (
    calculate_zscore_anomaly(
        X_train
    )
)

z_val_pred, z_val_score = (
    calculate_zscore_anomaly(
        X_val
    )
)

z_test_pred, z_test_score = (
    calculate_zscore_anomaly(
        X_test
    )
)


print(
    "Threshold: |Z| >= 3.0"
)

print(
    f"Test anomalies detected: "
    f"{z_test_pred.sum():,}"
)


# ============================================================
# 8. IQR DETECTOR
# ============================================================

print()
print("=" * 70)
print("8. IQR ANOMALY DETECTOR")
print("=" * 70)


Q1 = X_train_normal.quantile(
    0.25
)

Q3 = X_train_normal.quantile(
    0.75
)

IQR = Q3 - Q1


lower_bound = (
    Q1 - 1.5 * IQR
)

upper_bound = (
    Q3 + 1.5 * IQR
)


def calculate_iqr_anomaly(X):

    below = X.lt(
        lower_bound
    )

    above = X.gt(
        upper_bound
    )

    outlier_count = (
        below | above
    ).sum(axis=1)

    prediction = (
        outlier_count >= 1
    ).astype(int)

    return prediction, outlier_count


iqr_train_pred, iqr_train_score = (
    calculate_iqr_anomaly(
        X_train
    )
)

iqr_val_pred, iqr_val_score = (
    calculate_iqr_anomaly(
        X_val
    )
)

iqr_test_pred, iqr_test_score = (
    calculate_iqr_anomaly(
        X_test
    )
)


print(
    "Threshold: 1.5 × IQR"
)

print(
    f"Test anomalies detected: "
    f"{iqr_test_pred.sum():,}"
)


# ============================================================
# 9. ISOLATION FOREST
# ============================================================

print()
print("=" * 70)
print("9. ISOLATION FOREST")
print("=" * 70)


isolation_forest = IsolationForest(
    n_estimators=300,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)


print(
    "Training Isolation Forest..."
)


isolation_forest.fit(
    X_train_normal
)


# sklearn:
# +1 = normal
# -1 = anomaly

if_train_raw = (
    isolation_forest.predict(
        X_train
    )
)

if_val_raw = (
    isolation_forest.predict(
        X_val
    )
)

if_test_raw = (
    isolation_forest.predict(
        X_test
    )
)


if_train_pred = (
    if_train_raw == -1
).astype(int)

if_val_pred = (
    if_val_raw == -1
).astype(int)

if_test_pred = (
    if_test_raw == -1
).astype(int)


# Isolation Forest decision score
# Higher negative values indicate
# stronger anomaly behavior.

if_train_score = (
    -isolation_forest
    .decision_function(X_train)
)

if_val_score = (
    -isolation_forest
    .decision_function(X_val)
)

if_test_score = (
    -isolation_forest
    .decision_function(X_test)
)


print(
    f"Test anomalies detected: "
    f"{if_test_pred.sum():,}"
)


# ============================================================
# 10. EVALUATION FUNCTION
# ============================================================

print()
print("=" * 70)
print("10. MODEL EVALUATION")
print("=" * 70)


def evaluate_model(
    name,
    y_true,
    y_pred
):

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()


    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )


    print()
    print(
        f"--- {name} ---"
    )

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
        f"{false_positive_rate:.4f}"
    )

    print(
        f"True Negatives     : "
        f"{tn:,}"
    )

    print(
        f"False Positives    : "
        f"{fp:,}"
    )

    print(
        f"False Negatives    : "
        f"{fn:,}"
    )

    print(
        f"True Positives     : "
        f"{tp:,}"
    )


    return {
        "model": name,
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1": float(
            f1
        ),
        "false_positive_rate": float(
            false_positive_rate
        ),
        "true_negatives": int(
            tn
        ),
        "false_positives": int(
            fp
        ),
        "false_negatives": int(
            fn
        ),
        "true_positives": int(
            tp
        ),
    }


# ============================================================
# 11. TEST EVALUATION
# ============================================================

print()
print("=" * 70)
print("11. TEST SET EVALUATION")
print("=" * 70)


y_test = test_df[
    TARGET
].values


results = []


results.append(
    evaluate_model(
        "Z-Score",
        y_test,
        z_test_pred
    )
)


results.append(
    evaluate_model(
        "IQR",
        y_test,
        iqr_test_pred
    )
)


results.append(
    evaluate_model(
        "Isolation Forest",
        y_test,
        if_test_pred
    )
)


# ============================================================
# 12. SAVE MODEL RESULTS
# ============================================================

print()
print("=" * 70)
print("12. SAVING MODEL RESULTS")
print("=" * 70)


results_df = pd.DataFrame(
    results
)


results_file = os.path.join(
    OUTPUT_DIR,
    "baseline_model_results.csv"
)


results_df.to_csv(
    results_file,
    index=False
)


print(
    f"Saved: {results_file}"
)


# ============================================================
# 13. CREATE TEST PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("13. CREATING TEST PREDICTIONS")
print("=" * 70)


prediction_df = test_df[
    [
        "timestamp",
        "station",
        "T2M",
        "RH2M",
        "PS",
        TARGET,
    ]
].copy()


prediction_df.rename(
    columns={
        TARGET: "actual_anomaly"
    },
    inplace=True
)


# Z-Score
prediction_df[
    "zscore_anomaly"
] = z_test_pred

prediction_df[
    "zscore_score"
] = z_test_score


# IQR
prediction_df[
    "iqr_anomaly"
] = iqr_test_pred

prediction_df[
    "iqr_score"
] = iqr_test_score


# Isolation Forest
prediction_df[
    "isolation_forest_anomaly"
] = if_test_pred

prediction_df[
    "isolation_forest_score"
] = if_test_score


prediction_file = os.path.join(
    OUTPUT_DIR,
    "baseline_predictions_test.csv"
)


prediction_df.to_csv(
    prediction_file,
    index=False
)


print(
    f"Saved: {prediction_file}"
)


# ============================================================
# 14. SAVE ISOLATION FOREST MODEL
# ============================================================

print()
print("=" * 70)
print("14. SAVING ISOLATION FOREST MODEL")
print("=" * 70)


model_file = os.path.join(
    MODEL_DIR,
    "isolation_forest_baseline.joblib"
)


joblib.dump(
    isolation_forest,
    model_file
)


print(
    f"Saved: {model_file}"
)


# ============================================================
# 15. SAVE STATISTICS
# ============================================================

print()
print("=" * 70)
print("15. SAVING BASELINE STATISTICS")
print("=" * 70)


stats_file = os.path.join(
    MODEL_DIR,
    "baseline_statistics.joblib"
)


joblib.dump(
    {
        "features": FEATURES,
        "z_mean": z_mean,
        "z_std": z_std,
        "iqr_q1": Q1,
        "iqr_q3": Q3,
        "iqr_lower": lower_bound,
        "iqr_upper": upper_bound,
        "train_medians": train_medians,
    },
    stats_file
)


print(
    f"Saved: {stats_file}"
)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("16. BASELINE ML SUMMARY")
print("=" * 70)


print()
print(
    results_df.to_string(
        index=False
    )
)


print()
print("Output files:")

print(
    "  ✓ "
    "data/processed/baseline/"
    "baseline_model_results.csv"
)

print(
    "  ✓ "
    "data/processed/baseline/"
    "baseline_predictions_test.csv"
)

print(
    "  ✓ "
    "models/"
    "isolation_forest_baseline.joblib"
)

print(
    "  ✓ "
    "models/"
    "baseline_statistics.joblib"
)


print()
print("╔══════════════════════════════════════════════════════════════╗")
print("║             M6 BASELINE ML COMPLETE                        ║")
print("║                    STATUS: SUCCESS                         ║")
print("╚══════════════════════════════════════════════════════════════╝")
print()