import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ============================================================
# ATMOS — M7 MULTIVARIATE AUTOENCODER
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
    "autoencoder"
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
# REPRODUCIBILITY
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


print()

print(
    "╔══════════════════════════════════════════════════════════════╗"
)

print(
    "║              ATMOS AUTOENCODER ENGINE                      ║"
)

print(
    "║        M7 Multivariate Deep Anomaly Detection              ║"
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
    f"Loaded {len(df):,} records"
)


# ============================================================
# 2. DEFINE TARGET
# ============================================================

print()
print("=" * 70)
print("2. DEFINING GROUND-TRUTH TARGET")
print("=" * 70)

TARGET = "any_anomaly"

if TARGET not in df.columns:
    raise ValueError(
        f"Missing target column: {TARGET}"
    )

df[TARGET] = (
    pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )
    .fillna(0)
    .astype(int)
)

print(
    f"Normal records : "
    f"{(df[TARGET] == 0).sum():,}"
)

print(
    f"Anomaly records: "
    f"{(df[TARGET] == 1).sum():,}"
)


# ============================================================
# 3. SELECT FEATURES
# ============================================================

print()
print("=" * 70)
print("3. SELECTING AUTOENCODER FEATURES")
print("=" * 70)


FEATURES = [

    # Raw sensors
    "T2M",
    "RH2M",
    "PS",

    # Rate of change
    "T2M_diff",
    "RH2M_diff",
    "PS_diff",

    "T2M_pct_change",
    "RH2M_pct_change",
    "PS_pct_change",

    # Rolling behavior
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


missing = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing:
    raise ValueError(
        f"Missing features: {missing}"
    )


print(
    f"Selected {len(FEATURES)} features"
)

for feature in FEATURES:
    print(
        f"  ✓ {feature}"
    )


# ============================================================
# 4. TRAIN / VALIDATION / TEST
# ============================================================

print()
print("=" * 70)
print("4. DATA SPLIT")
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
    f"Training   : {len(train_df):,}"
)

print(
    f"Validation : {len(val_df):,}"
)

print(
    f"Testing    : {len(test_df):,}"
)


# ============================================================
# 5. NORMAL TRAINING DATA
# ============================================================

print()
print("=" * 70)
print("5. PREPARING NORMAL TRAINING DATA")
print("=" * 70)


normal_train_df = train_df[
    train_df[TARGET] == 0
].copy()


print(
    f"Normal training records: "
    f"{len(normal_train_df):,}"
)

print(
    f"Excluded anomaly records: "
    f"{len(train_df) - len(normal_train_df):,}"
)


# ============================================================
# 6. CREATE FEATURE MATRICES
# ============================================================

print()
print("=" * 70)
print("6. CREATING FEATURE MATRICES")
print("=" * 70)


X_train = normal_train_df[
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


# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================

print()
print("=" * 70)
print("7. HANDLING MISSING VALUES")
print("=" * 70)


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
    "Missing values handled "
    "using training medians."
)


# ============================================================
# 8. STANDARDIZATION
# ============================================================

print()
print("=" * 70)
print("8. FEATURE STANDARDIZATION")
print("=" * 70)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_val_scaled = scaler.transform(
    X_val
)

X_test_scaled = scaler.transform(
    X_test
)


print(
    f"Training matrix: "
    f"{X_train_scaled.shape}"
)

print(
    f"Validation matrix: "
    f"{X_val_scaled.shape}"
)

print(
    f"Test matrix: "
    f"{X_test_scaled.shape}"
)


# ============================================================
# 9. BUILD AUTOENCODER
# ============================================================

print()
print("=" * 70)
print("9. BUILDING AUTOENCODER")
print("=" * 70)


input_dim = X_train_scaled.shape[1]


inputs = keras.Input(
    shape=(input_dim,),
    name="weather_features"
)


# ------------------------------------------------------------
# Encoder
# ------------------------------------------------------------

x = layers.Dense(
    32,
    activation="relu"
)(inputs)

x = layers.Dropout(
    0.10
)(x)

x = layers.Dense(
    16,
    activation="relu"
)(x)

latent = layers.Dense(
    8,
    activation="relu",
    name="latent_space"
)(x)


# ------------------------------------------------------------
# Decoder
# ------------------------------------------------------------

x = layers.Dense(
    16,
    activation="relu"
)(latent)

x = layers.Dense(
    32,
    activation="relu"
)(x)

outputs = layers.Dense(
    input_dim,
    activation="linear",
    name="reconstruction"
)(x)


autoencoder = keras.Model(
    inputs,
    outputs,
    name="ATMOS_Autoencoder"
)


autoencoder.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse"
)


print()
autoencoder.summary()


# ============================================================
# 10. CALLBACKS
# ============================================================

print()
print("=" * 70)
print("10. CONFIGURING TRAINING")
print("=" * 70)


model_file = os.path.join(
    MODEL_DIR,
    "atmos_autoencoder.keras"
)


callbacks = [

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    ),

    keras.callbacks.ModelCheckpoint(
        model_file,
        monitor="val_loss",
        save_best_only=True
    ),

]


# ============================================================
# 11. TRAIN AUTOENCODER
# ============================================================

print()
print("=" * 70)
print("11. TRAINING AUTOENCODER")
print("=" * 70)

print()
print(
    "Training model..."
)

print(
    "This may take a few minutes."
)


history = autoencoder.fit(

    X_train_scaled,

    X_train_scaled,

    validation_data=(
        X_val_scaled,
        X_val_scaled
    ),

    epochs=100,

    batch_size=128,

    shuffle=True,

    callbacks=callbacks,

    verbose=1
)


print()
print(
    "Training complete."
)


# ============================================================
# 12. CALCULATE RECONSTRUCTION ERRORS
# ============================================================

print()
print("=" * 70)
print("12. CALCULATING RECONSTRUCTION ERRORS")
print("=" * 70)


train_reconstructed = (
    autoencoder.predict(
        X_train_scaled,
        verbose=0
    )
)

val_reconstructed = (
    autoencoder.predict(
        X_val_scaled,
        verbose=0
    )
)

test_reconstructed = (
    autoencoder.predict(
        X_test_scaled,
        verbose=0
    )
)


train_error = np.mean(
    np.square(
        X_train_scaled -
        train_reconstructed
    ),
    axis=1
)

val_error = np.mean(
    np.square(
        X_val_scaled -
        val_reconstructed
    ),
    axis=1
)

test_error = np.mean(
    np.square(
        X_test_scaled -
        test_reconstructed
    ),
    axis=1
)


print(
    f"Training error mean: "
    f"{train_error.mean():.6f}"
)

print(
    f"Validation error mean: "
    f"{val_error.mean():.6f}"
)

print(
    f"Test error mean: "
    f"{test_error.mean():.6f}"
)


# ============================================================
# 13. DETERMINE ANOMALY THRESHOLD
# ============================================================

print()
print("=" * 70)
print("13. DETERMINING ANOMALY THRESHOLD")
print("=" * 70)


# Use normal training reconstruction errors.
# 99th percentile provides a conservative baseline
# for reconstruction-error anomalies.

threshold = np.percentile(
    train_error,
    99
)


print(
    f"Anomaly threshold: "
    f"{threshold:.6f}"
)


# ============================================================
# 14. GENERATE PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("14. GENERATING ANOMALY PREDICTIONS")
print("=" * 70)


train_pred = (
    train_error >= threshold
).astype(int)

val_pred = (
    val_error >= threshold
).astype(int)

test_pred = (
    test_error >= threshold
).astype(int)


print(
    f"Test anomalies detected: "
    f"{test_pred.sum():,}"
)


# ============================================================
# 15. EVALUATION
# ============================================================

print()
print("=" * 70)
print("15. AUTOENCODER EVALUATION")
print("=" * 70)


y_test = test_df[
    TARGET
].values


precision = precision_score(
    y_test,
    test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_pred,
    zero_division=0
)


cm = confusion_matrix(
    y_test,
    test_pred,
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


# ============================================================
# 16. SAVE MODEL METRICS
# ============================================================

print()
print("=" * 70)
print("16. SAVING AUTOENCODER METRICS")
print("=" * 70)


metrics = {

    "model": "Multivariate Autoencoder",

    "features": FEATURES,

    "input_dimension": int(
        input_dim
    ),

    "latent_dimension": 8,

    "threshold": float(
        threshold
    ),

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


metrics_file = os.path.join(
    OUTPUT_DIR,
    "autoencoder_metrics.json"
)


with open(
    metrics_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


print(
    f"Saved: {metrics_file}"
)


# ============================================================
# 17. SAVE SCALER
# ============================================================

print()
print("=" * 70)
print("17. SAVING FEATURE SCALER")
print("=" * 70)


scaler_file = os.path.join(
    MODEL_DIR,
    "autoencoder_scaler.joblib"
)


joblib.dump(
    scaler,
    scaler_file
)


print(
    f"Saved: {scaler_file}"
)


# ============================================================
# 18. SAVE RECONSTRUCTION THRESHOLD
# ============================================================

threshold_file = os.path.join(
    MODEL_DIR,
    "autoencoder_threshold.json"
)


with open(
    threshold_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "threshold": float(
                threshold
            ),
            "percentile": 99
        },
        f,
        indent=4
    )


print(
    f"Saved: {threshold_file}"
)


# ============================================================
# 19. SAVE TEST PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("19. SAVING TEST PREDICTIONS")
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


prediction_df[
    "autoencoder_anomaly"
] = test_pred


prediction_df[
    "reconstruction_error"
] = test_error


prediction_file = os.path.join(
    OUTPUT_DIR,
    "autoencoder_predictions_test.csv"
)


prediction_df.to_csv(
    prediction_file,
    index=False
)


print(
    f"Saved: {prediction_file}"
)


# ============================================================
# 20. SAVE TRAINING HISTORY
# ============================================================

history_file = os.path.join(
    OUTPUT_DIR,
    "training_history.csv"
)


history_df = pd.DataFrame(
    history.history
)


history_df.to_csv(
    history_file,
    index=False
)


print(
    f"Saved: {history_file}"
)


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("21. M7 AUTOENCODER SUMMARY")
print("=" * 70)


print()
print(
    "Model architecture:"
)

print(
    "22 → 32 → 16 → 8 → "
    "16 → 32 → 22"
)


print()
print(
    "Performance:"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"FPR       : {false_positive_rate:.4f}"
)


print()
print(
    "Output files:"
)

print(
    "  ✓ models/atmos_autoencoder.keras"
)

print(
    "  ✓ models/autoencoder_scaler.joblib"
)

print(
    "  ✓ models/autoencoder_threshold.json"
)

print(
    "  ✓ data/processed/autoencoder/"
    "autoencoder_metrics.json"
)

print(
    "  ✓ data/processed/autoencoder/"
    "autoencoder_predictions_test.csv"
)

print(
    "  ✓ data/processed/autoencoder/"
    "training_history.csv"
)


print()

print(
    "╔══════════════════════════════════════════════════════════════╗"
)

print(
    "║             M7 AUTOENCODER COMPLETE                        ║"
)

print(
    "║                    STATUS: SUCCESS                         ║"
)

print(
    "╚══════════════════════════════════════════════════════════════╝"
)

print()