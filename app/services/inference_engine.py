from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODELS_DIR = BASE_DIR / "models"


# ============================================================
# MODEL FILES
# ============================================================

AUTOENCODER_PATH = MODELS_DIR / "atmos_autoencoder.keras"
SCALER_PATH = MODELS_DIR / "autoencoder_scaler.joblib"
THRESHOLD_PATH = MODELS_DIR / "autoencoder_threshold.json"

T2M_MODEL_PATH = MODELS_DIR / "t2m_fault_classifier.joblib"
RH2M_MODEL_PATH = MODELS_DIR / "rh2m_fault_classifier.joblib"
PS_MODEL_PATH = MODELS_DIR / "ps_fault_classifier.joblib"


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "T2M",
    "RH2M",
    "PS",

    "T2M_diff",
    "RH2M_diff",
    "PS_diff",

    "T2M_pct_change",
    "RH2M_pct_change",
    "PS_pct_change",

    "T2M_rolling_mean_6",
    "RH2M_rolling_mean_6",
    "PS_rolling_mean_6",

    "T2M_rolling_std_6",
    "RH2M_rolling_std_6",
    "PS_rolling_std_6",

    "T2M_local_z",
    "RH2M_local_z",
    "PS_local_z",

    "T2M_RH2M_ratio",
    "T2M_PS_ratio",

    "T2M_RH2M_difference",
    "T2M_PS_difference",
]


# ============================================================
# SENSOR CONFIGURATION
# ============================================================

SENSOR_CONFIG = {
    "T2M": {
        "model_path": T2M_MODEL_PATH,
        "fault_column": "temperature_fault_type",
    },
    "RH2M": {
        "model_path": RH2M_MODEL_PATH,
        "fault_column": "humidity_fault_type",
    },
    "PS": {
        "model_path": PS_MODEL_PATH,
        "fault_column": "pressure_fault_type",
    },
}


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading SkyGuard inference models...")


autoencoder = tf.keras.models.load_model(
    AUTOENCODER_PATH,
    compile=False
)

scaler = joblib.load(SCALER_PATH)


with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
    threshold_data = json.load(f)


AUTOENCODER_THRESHOLD = float(
    threshold_data["threshold"]
)


fault_models = {}

for sensor_name, config in SENSOR_CONFIG.items():

    fault_models[sensor_name] = joblib.load(
        config["model_path"]
    )


print("SkyGuard inference models loaded successfully.")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value, default=0.0):
    """
    Safely convert a value to float.
    """

    try:

        if value is None:
            return default

        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except (ValueError, TypeError):

        return default


# ------------------------------------------------------------
# Severity
# ------------------------------------------------------------

def severity_from_fault(
    fault,
    confidence,
    local_z=0.0
):
    """
    Determine severity using classifier result
    and statistical deviation.
    """

    fault = str(fault).lower()

    confidence = safe_float(confidence)

    local_z = abs(
        safe_float(local_z)
    )

    # Normal
    if fault == "normal":

        if local_z >= 4:
            return "CRITICAL"

        if local_z >= 3:
            return "HIGH"

        return "NORMAL"

    # Very strong anomaly
    if local_z >= 4 or confidence >= 95:

        if fault in {
            "spike",
            "drop"
        }:
            return "CRITICAL"

    # Strong sudden faults
    if fault in {
        "spike",
        "drop"
    }:

        if confidence >= 85:
            return "HIGH"

        if confidence >= 60:
            return "MEDIUM"

        return "LOW"

    # Drift / stuck
    if fault in {
        "drift",
        "stuck"
    }:

        if confidence >= 90:
            return "HIGH"

        if confidence >= 60:
            return "MEDIUM"

        return "LOW"

    # Unknown fault
    if confidence >= 90:
        return "HIGH"

    if confidence >= 60:
        return "MEDIUM"

    return "LOW"


# ------------------------------------------------------------
# Prediction confidence
# ------------------------------------------------------------

def get_prediction(
    model,
    features
):
    """
    Run classifier prediction and return:
    fault + confidence.
    """

    probabilities = model.predict_proba(
        features
    )[0]

    classes = model.classes_

    best_index = int(
        np.argmax(probabilities)
    )

    fault = str(
        classes[best_index]
    ).lower()

    confidence = float(
        probabilities[best_index] * 100
    )

    return fault, confidence


# ============================================================
# MAIN INFERENCE FUNCTION
# ============================================================

def run_inference(record):
    """
    Run complete live inference for one AWS observation.

    Pipeline:

        Observation
            ↓
        Feature extraction
            ↓
        Autoencoder
            ↓
        Fault classifiers
            ↓
        Statistical anomaly detection
            ↓
        Decision fusion
            ↓
        Primary diagnosis
    """

    if record is None:

        raise ValueError(
            "No observation supplied for inference."
        )


    # ========================================================
    # BASIC INFORMATION
    # ========================================================

    timestamp = record.get(
        "timestamp"
    )

    station = record.get(
        "station",
        "unknown"
    )


    # ========================================================
    # FEATURE VECTOR
    # ========================================================

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in record
    ]

    if missing_features:

        raise ValueError(
            "Missing inference features: "
            + ", ".join(missing_features)
        )


    feature_values = []

    for feature in FEATURE_COLUMNS:

        feature_values.append(
            safe_float(
                record.get(feature)
            )
        )


    X = np.array(
        [feature_values],
        dtype=np.float32
    )


    # ========================================================
    # AUTOENCODER
    # ========================================================

    X_scaled = scaler.transform(
        X
    )


    reconstruction = autoencoder.predict(
        X_scaled,
        verbose=0
    )


    reconstruction_error = float(
        np.mean(
            np.square(
                X_scaled - reconstruction
            ),
            axis=1
        )[0]
    )


    autoencoder_anomaly = (
        reconstruction_error
        >= AUTOENCODER_THRESHOLD
    )


    # Convert reconstruction error
    # into a relative anomaly score.
    autoencoder_score = (
        reconstruction_error
        / max(AUTOENCODER_THRESHOLD, 1e-9)
    )


    # Keep score within reasonable range.
    autoencoder_score = float(
        min(autoencoder_score, 1.0)
    )


    # ========================================================
    # SENSOR-LEVEL CLASSIFICATION
    # ========================================================

    sensor_results = {}


    for sensor_name, model in fault_models.items():

        fault, confidence = get_prediction(
            model,
            X
        )


        # --------------------------------------------
        # Local Z-score
        # --------------------------------------------

        z_column = (
            f"{sensor_name}_local_z"
        )

        local_z = safe_float(
            record.get(z_column)
        )


        abs_z = abs(
            local_z
        )


        # --------------------------------------------
        # Severity
        # --------------------------------------------

        severity = severity_from_fault(
            fault=fault,
            confidence=confidence,
            local_z=local_z
        )


        # --------------------------------------------
        # Strong classifier fault
        # --------------------------------------------

        strong_fault = (
            fault != "normal"
            and confidence >= 60
        )


        # --------------------------------------------
        # Strong statistical deviation
        # --------------------------------------------

        strong_statistical_deviation = (
            abs_z >= 3
        )


        # --------------------------------------------
        # Autoencoder anomaly
        #
        # Autoencoder is multivariate, so it should
        # not automatically assign the anomaly to
        # every sensor.
        # --------------------------------------------

        sensor_anomaly = (
            strong_fault
            or strong_statistical_deviation
        )


        # If a strong statistical anomaly exists,
        # make sure severity isn't NORMAL.
        if strong_statistical_deviation:

            if abs_z >= 4:

                severity = "CRITICAL"

            elif abs_z >= 3:

                severity = "HIGH"


        sensor_results[sensor_name] = {

            "fault": fault,

            "confidence": round(
                confidence,
                2
            ),

            "local_z": round(
                local_z,
                4
            ),

            "anomaly": bool(
                sensor_anomaly
            ),

            "severity": severity,
        }


    # ========================================================
    # OVERALL ANOMALY DECISION
    # ========================================================

    sensor_anomaly_detected = any(
        result["anomaly"]
        for result in sensor_results.values()
    )


    overall_anomaly = (
        sensor_anomaly_detected
        or autoencoder_anomaly
    )


    status = (
        "ANOMALY"
        if overall_anomaly
        else "NORMAL"
    )


    # ========================================================
    # PRIMARY DIAGNOSIS
    # ========================================================
    #
    # IMPORTANT FIX:
    #
    # A classifier predicting "normal" must NOT override
    # a statistically detected anomaly.
    #
    # Priority:
    #
    # 1. Confirmed classifier fault
    # 2. Strong statistical anomaly
    # 3. Multivariate autoencoder anomaly
    # 4. Normal
    #
    # ========================================================

    candidate_diagnoses = []


    for sensor_name, result in sensor_results.items():

        fault = result["fault"]

        confidence = safe_float(
            result["confidence"]
        )

        local_z = safe_float(
            result["local_z"]
        )

        anomaly = bool(
            result["anomaly"]
        )


        # ----------------------------------------------------
        # CASE 1:
        # Classifier identified an actual fault
        # ----------------------------------------------------

        if (
            fault != "normal"
            and confidence >= 60
            and anomaly
        ):

            candidate_diagnoses.append({

                "sensor": sensor_name,

                "fault": fault,

                "confidence": round(
                    confidence,
                    2
                ),

                "severity": result[
                    "severity"
                ],

                "priority": 3,

            })


        # ----------------------------------------------------
        # CASE 2:
        # Statistical anomaly
        #
        # Example:
        #
        # classifier = normal
        # confidence = 99%
        # local_z = 3.05
        #
        # This MUST still become a primary diagnosis.
        # ----------------------------------------------------

        elif (
            anomaly
            and abs(local_z) >= 3
        ):

            # Convert z-score into an evidence confidence.
            #
            # z = 3 → 100%
            # z > 3 → capped at 100%
            #
            statistical_confidence = min(
                abs(local_z) / 3 * 100,
                100
            )


            candidate_diagnoses.append({

                "sensor": sensor_name,

                "fault": "statistical_anomaly",

                "confidence": round(
                    statistical_confidence,
                    2
                ),

                "severity": result[
                    "severity"
                ],

                "priority": 2,

            })


    # ========================================================
    # SELECT STRONGEST SENSOR DIAGNOSIS
    # ========================================================

    if candidate_diagnoses:

        candidate_diagnoses.sort(
            key=lambda item: (
                item["priority"],
                item["confidence"],
                abs(
                    sensor_results[
                        item["sensor"]
                    ]["local_z"]
                )
            ),
            reverse=True
        )


        selected = (
            candidate_diagnoses[0]
        )


        primary_diagnosis = {

            "sensor": selected[
                "sensor"
            ],

            "fault": selected[
                "fault"
            ],

            "confidence": selected[
                "confidence"
            ],

            "severity": selected[
                "severity"
            ],

        }


    # ========================================================
    # AUTOENCODER-ONLY ANOMALY
    # ========================================================

    elif autoencoder_anomaly:

        primary_diagnosis = {

            "sensor": None,

            "fault": "multivariate_anomaly",

            "confidence": round(
                autoencoder_score * 100,
                2
            ),

            "severity": "HIGH",

        }


    # ========================================================
    # NORMAL
    # ========================================================

    else:

        primary_diagnosis = {

            "sensor": None,

            "fault": "normal",

            "confidence": 0,

            "severity": "NORMAL",

        }


    # ========================================================
    # FINAL RESULT
    # ========================================================

    result = {

        "timestamp": timestamp,

        "station": station,

        "status": status,

        "primary_diagnosis": primary_diagnosis,

        "autoencoder": {

            "anomaly": bool(
                autoencoder_anomaly
            ),

            "score": round(
                autoencoder_score,
                4
            ),

            "reconstruction_error": round(
                reconstruction_error,
                6
            ),

            "threshold": round(
                AUTOENCODER_THRESHOLD,
                6
            ),

        },

        "sensors": sensor_results,

    }


    return result