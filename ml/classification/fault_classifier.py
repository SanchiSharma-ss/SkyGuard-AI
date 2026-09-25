import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# ATMOS — M9 FAULT CLASSIFICATION ENGINE
#
# Predicts:
#   normal
#   spike
#   drift
#   stuck
#   drop
#
# IMPORTANT:
# Fault/anomaly labels are targets only.
# They are NEVER used as input features.
# ============================================================


BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
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
    "classification"
)


MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    MODEL_DIR,
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
    "║              ATMOS FAULT CLASSIFIER                        ║"
)

print(
    "║          M9 Sensor Fault Classification                    ║"
)

print(
    "╚══════════════════════════════════════════════════════════════╝"
)

print()


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("1. LOADING FEATURE DATA")
print("=" * 70)


df = pd.read_csv(
    INPUT_FILE
)


df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)


print(
    f"Loaded records: "
    f"{len(df):,}"
)


# ============================================================
# 2. DEFINE INPUT FEATURES
# ============================================================

print()
print("=" * 70)
print("2. SELECTING CLASSIFICATION FEATURES")
print("=" * 70)


FEATURES = [

    # --------------------------------------------------------
    # Raw sensor measurements
    # --------------------------------------------------------

    "T2M",
    "RH2M",
    "PS",

    # --------------------------------------------------------
    # Temporal changes
    # --------------------------------------------------------

    "T2M_diff",
    "RH2M_diff",
    "PS_diff",

    "T2M_pct_change",
    "RH2M_pct_change",
    "PS_pct_change",

    # --------------------------------------------------------
    # Rolling behavior
    # --------------------------------------------------------

    "T2M_rolling_mean_6",
    "RH2M_rolling_mean_6",
    "PS_rolling_mean_6",

    "T2M_rolling_std_6",
    "RH2M_rolling_std_6",
    "PS_rolling_std_6",

    # --------------------------------------------------------
    # Local deviation
    # --------------------------------------------------------

    "T2M_local_z",
    "RH2M_local_z",
    "PS_local_z",

    # --------------------------------------------------------
    # Cross-sensor relationships
    # --------------------------------------------------------

    "T2M_RH2M_ratio",
    "T2M_PS_ratio",

    "T2M_RH2M_difference",
    "T2M_PS_difference",

]


missing_features = [

    feature

    for feature in FEATURES

    if feature not in df.columns

]


if missing_features:

    raise ValueError(
        f"Missing features: {missing_features}"
    )


print(
    f"Selected features: "
    f"{len(FEATURES)}"
)


for feature in FEATURES:

    print(
        f"  ✓ {feature}"
    )


# ============================================================
# 3. DEFINE FAULT LABELS
# ============================================================

print()
print("=" * 70)
print("3. VERIFYING FAULT LABELS")
print("=" * 70)


FAULT_LABELS = {

    "T2M": "temperature_fault_type",

    "RH2M": "humidity_fault_type",

    "PS": "pressure_fault_type",

}


for sensor, label in FAULT_LABELS.items():

    if label not in df.columns:

        raise ValueError(
            f"Missing fault label: {label}"
        )


print(
    "Fault labels verified:"
)

for sensor, label in FAULT_LABELS.items():

    print(
        f"  ✓ {sensor} → {label}"
    )


# ============================================================
# 4. DISPLAY FAULT DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("4. FAULT DISTRIBUTION")
print("=" * 70)


for sensor, label in FAULT_LABELS.items():

    print()
    print(
        f"{sensor} fault distribution:"
    )

    print(
        df[label]
        .value_counts()
        .to_string()
    )


# ============================================================
# 5. DATA SPLIT
# ============================================================

print()
print("=" * 70)
print("5. DATA SPLIT")
print("=" * 70)


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
# 6. PREPARE FEATURES
# ============================================================

print()
print("=" * 70)
print("6. PREPARING FEATURES")
print("=" * 70)


X_train = train_df[
    FEATURES
].copy()


X_val = val_df[
    FEATURES
].copy()


X_test = test_df[
    FEATURES
].copy()


# Replace infinity
X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)

X_val = X_val.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)


# Training medians only
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
    "Missing values handled."
)


# ============================================================
# 7. TRAIN ONE CLASSIFIER PER SENSOR
# ============================================================

print()
print("=" * 70)
print("7. TRAINING SENSOR-SPECIFIC CLASSIFIERS")
print("=" * 70)


models = {}


evaluation_results = []


prediction_outputs = test_df[
    [
        "timestamp",
        "station",
        "T2M",
        "RH2M",
        "PS",
    ]
].copy()


# ============================================================
# SENSOR LOOP
# ============================================================

for sensor, label_column in FAULT_LABELS.items():

    print()
    print(
        "-" * 70
    )

    print(
        f"TRAINING CLASSIFIER: {sensor}"
    )

    print(
        "-" * 70
    )


    # --------------------------------------------------------
    # TARGETS
    # --------------------------------------------------------

    y_train = (
        train_df[
            label_column
        ]
        .fillna("normal")
        .astype(str)
    )


    y_val = (
        val_df[
            label_column
        ]
        .fillna("normal")
        .astype(str)
    )


    y_test = (
        test_df[
            label_column
        ]
        .fillna("normal")
        .astype(str)
    )


    print()
    print(
        "Training class distribution:"
    )

    print(
        y_train
        .value_counts()
        .to_string()
    )


    # --------------------------------------------------------
    # CLASSIFIER
    # --------------------------------------------------------

    classifier = RandomForestClassifier(

        n_estimators=300,

        max_depth=18,

        min_samples_leaf=2,

        class_weight="balanced_subsample",

        random_state=42,

        n_jobs=-1,

    )


    print()
    print(
        "Training Random Forest..."
    )


    classifier.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_pred = classifier.predict(
        X_val
    )


    val_f1 = f1_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0
    )


    print(
        f"Validation Macro F1: "
        f"{val_f1:.4f}"
    )


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_pred = classifier.predict(
        X_test
    )


    test_probabilities = (
        classifier.predict_proba(
            X_test
        )
    )


    # Highest probability
    test_confidence = (
        np.max(
            test_probabilities,
            axis=1
        )
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        test_pred
    )


    precision = precision_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0
    )


    recall = recall_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0
    )


    print()
    print(
        f"{sensor} TEST PERFORMANCE"
    )

    print(
        f"Accuracy           : "
        f"{accuracy:.4f}"
    )

    print(
        f"Macro Precision    : "
        f"{precision:.4f}"
    )

    print(
        f"Macro Recall       : "
        f"{recall:.4f}"
    )

    print(
        f"Macro F1           : "
        f"{f1:.4f}"
    )


    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    print()
    print(
        "Classification Report:"
    )

    print(
        classification_report(
            y_test,
            test_pred,
            zero_division=0
        )
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    labels = sorted(
        y_test.unique()
    )


    cm = confusion_matrix(
        y_test,
        test_pred,
        labels=labels
    )


    cm_df = pd.DataFrame(
        cm,
        index=[
            f"actual_{x}"
            for x in labels
        ],
        columns=[
            f"predicted_{x}"
            for x in labels
        ]
    )


    cm_file = os.path.join(

        OUTPUT_DIR,

        f"{sensor.lower()}_confusion_matrix.csv"

    )


    cm_df.to_csv(
        cm_file
    )


    print(
        f"Confusion matrix saved: "
        f"{cm_file}"
    )


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance_df = pd.DataFrame({

        "feature":
            FEATURES,

        "importance":
            classifier.feature_importances_

    })


    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
    )


    importance_file = os.path.join(

        OUTPUT_DIR,

        f"{sensor.lower()}_feature_importance.csv"

    )


    importance_df.to_csv(
        importance_file,
        index=False
    )


    print(
        "Top features:"
    )

    print(
        importance_df
        .head(10)
        .to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_file = os.path.join(

        MODEL_DIR,

        f"{sensor.lower()}_fault_classifier.joblib"

    )


    joblib.dump(
        classifier,
        model_file
    )


    print(
        f"Model saved: "
        f"{model_file}"
    )


    # --------------------------------------------------------
    # SAVE MODEL OBJECT
    # --------------------------------------------------------

    models[sensor] = classifier


    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    evaluation_results.append({

        "sensor":
            sensor,

        "accuracy":
            float(accuracy),

        "macro_precision":
            float(precision),

        "macro_recall":
            float(recall),

        "macro_f1":
            float(f1),

        "validation_macro_f1":
            float(val_f1),

        "classes":
            sorted(
                y_train.unique()
            ),

    })


    # --------------------------------------------------------
    # TEST PREDICTIONS
    # --------------------------------------------------------

    prediction_outputs[
        f"{sensor}_actual_fault"
    ] = y_test.values


    prediction_outputs[
        f"{sensor}_predicted_fault"
    ] = test_pred


    prediction_outputs[
        f"{sensor}_fault_confidence"
    ] = (
        test_confidence
        * 100
    )


# ============================================================
# 8. SAVE OVERALL RESULTS
# ============================================================

print()
print("=" * 70)
print("8. SAVING CLASSIFICATION RESULTS")
print("=" * 70)


results_df = pd.DataFrame(
    evaluation_results
)


results_file = os.path.join(
    OUTPUT_DIR,
    "fault_classifier_results.csv"
)


results_df.to_csv(
    results_file,
    index=False
)


print(
    f"Saved: {results_file}"
)


# ============================================================
# 9. SAVE TEST PREDICTIONS
# ============================================================

prediction_file = os.path.join(

    OUTPUT_DIR,

    "fault_predictions_test.csv"

)


prediction_outputs.to_csv(
    prediction_file,
    index=False
)


print(
    f"Saved: {prediction_file}"
)


# ============================================================
# 10. SAVE FEATURE LIST
# ============================================================

feature_config_file = os.path.join(

    MODEL_DIR,

    "fault_classifier_features.json"

)


with open(
    feature_config_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(

        {
            "features": FEATURES,

            "target_columns":
                FAULT_LABELS,

            "sensors":
                list(
                    FAULT_LABELS.keys()
                ),

            "model":
                "RandomForestClassifier",

            "n_estimators":
                300,

            "random_state":
                42,

        },

        f,

        indent=4

    )


print(
    f"Saved: {feature_config_file}"
)


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("11. M9 CLASSIFICATION SUMMARY")
print("=" * 70)


print()
print(
    results_df.to_string(
        index=False
    )
)


print()
print(
    "Generated models:"
)


for sensor in FAULT_LABELS:

    print(
        f"  ✓ models/"
        f"{sensor.lower()}_fault_classifier.joblib"
    )


print()
print(
    "Generated outputs:"
)

print(
    "  ✓ classification/"
    "fault_classifier_results.csv"
)

print(
    "  ✓ classification/"
    "fault_predictions_test.csv"
)

print(
    "  ✓ classification/"
    "t2m_confusion_matrix.csv"
)

print(
    "  ✓ classification/"
    "rh2m_confusion_matrix.csv"
)

print(
    "  ✓ classification/"
    "ps_confusion_matrix.csv"
)


print()

print(
    "╔══════════════════════════════════════════════════════════════╗"
)

print(
    "║             M9 FAULT CLASSIFICATION COMPLETE              ║"
)

print(
    "║                    STATUS: SUCCESS                         ║"
)

print(
    "╚══════════════════════════════════════════════════════════════╝"
)

print()