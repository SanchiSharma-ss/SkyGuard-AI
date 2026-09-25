import os
from datetime import datetime


# ============================================================
# ATMOS — LIVE HEALTH + ALERT ENGINE
# M12.2
# ============================================================


def calculate_live_health(
    sensor_result
):

    fault = str(
        sensor_result.get(
            "fault",
            "normal"
        )
    ).lower()

    confidence = float(
        sensor_result.get(
            "confidence",
            0
        )
    )

    local_z = abs(
        float(
            sensor_result.get(
                "local_z",
                0
            )
        )
    )

    anomaly = bool(
        sensor_result.get(
            "anomaly",
            False
        )
    )

    severity = str(
        sensor_result.get(
            "severity",
            "NORMAL"
        )
    ).upper()


    # --------------------------------------------------------
    # Base score
    # --------------------------------------------------------

    score = 100.0


    # --------------------------------------------------------
    # Fault penalty
    # --------------------------------------------------------

    fault_penalties = {

        "normal": 0,

        "drift": 12,

        "stuck": 18,

        "drop": 25,

        "spike": 25,

    }


    score -= fault_penalties.get(
        fault,
        10
    )


    # --------------------------------------------------------
    # Anomaly penalty
    # --------------------------------------------------------

    if anomaly:

        score -= 15


    # --------------------------------------------------------
    # Confidence penalty
    #
    # High-confidence faults have a stronger
    # effect on sensor health.
    # --------------------------------------------------------

    if fault != "normal":

        confidence_factor = (
            confidence / 100
        )

        score -= (
            confidence_factor
            * 15
        )


    # --------------------------------------------------------
    # Statistical deviation
    # --------------------------------------------------------

    if local_z >= 3:

        score -= 15

    elif local_z >= 2.5:

        score -= 8

    elif local_z >= 2:

        score -= 3


    # --------------------------------------------------------
    # Severity penalty
    # --------------------------------------------------------

    severity_penalties = {

        "NORMAL": 0,

        "LOW": 3,

        "MEDIUM": 8,

        "HIGH": 15,

        "CRITICAL": 25,

    }


    score -= severity_penalties.get(
        severity,
        0
    )


    score = max(
        0,
        min(
            100,
            score
        )
    )


    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    if score >= 90:

        status = "EXCELLENT"

    elif score >= 75:

        status = "HEALTHY"

    elif score >= 60:

        status = "WARNING"

    elif score >= 40:

        status = "DEGRADED"

    else:

        status = "CRITICAL"


    return {

        "score":
            round(
                score,
                2
            ),

        "status":
            status,

    }


# ============================================================
# ALERT GENERATION
# ============================================================

def generate_alert(
    inference_result
):

    station = inference_result.get(
        "station"
    )

    timestamp = inference_result.get(
        "timestamp"
    )

    overall_status = inference_result.get(
        "status",
        "NORMAL"
    )


    alerts = []


    sensors = inference_result.get(
        "sensors",
        {}
    )


    for sensor, result in sensors.items():

        if not result.get(
            "anomaly",
            False
        ):

            continue


        fault = result.get(
            "fault",
            "unknown"
        )


        confidence = float(
            result.get(
                "confidence",
                0
            )
        )


        severity = result.get(
            "severity",
            "MEDIUM"
        )


        health = calculate_live_health(
            result
        )


        # ----------------------------------------------------
        # Human-readable evidence
        # ----------------------------------------------------

        evidence = []


        if fault != "normal":

            evidence.append(
                f"Fault classifier: {fault}"
            )


        if confidence >= 80:

            evidence.append(
                "High classifier confidence"
            )

        elif confidence >= 60:

            evidence.append(
                "Moderate classifier confidence"
            )

        else:

            evidence.append(
                "Low classifier confidence"
            )


        local_z = abs(
            float(
                result.get(
                    "local_z",
                    0
                )
            )
        )


        if local_z >= 3:

            evidence.append(
                f"Strong statistical deviation "
                f"(local z-score={local_z:.2f})"
            )

        elif local_z >= 2:

            evidence.append(
                f"Elevated statistical deviation "
                f"(local z-score={local_z:.2f})"
            )


        alert = {

            "alert_id":
                f"{station}_{sensor}_{timestamp}",

            "timestamp":
                timestamp,

            "generated_at":
                datetime.now().isoformat(),

            "station":
                station,

            "sensor":
                sensor,

            "status":
                "ACTIVE",

            "fault":
                fault,

            "severity":
                severity,

            "confidence":
                confidence,

            "sensor_health":
                health["score"],

            "health_status":
                health["status"],

            "local_z":
                result.get(
                    "local_z"
                ),

            "evidence":
                evidence,

        }


        alerts.append(
            alert
        )


    return {

        "station":
            station,

        "timestamp":
            timestamp,

        "overall_status":
            overall_status,

        "alert_count":
            len(alerts),

        "alerts":
            alerts,

    }


# ============================================================
# COMPLETE LIVE RESULT
# ============================================================

def build_live_result(
    inference_result
):

    sensor_health = {}


    for sensor, result in (
        inference_result
        .get(
            "sensors",
            {}
        )
        .items()
    ):

        sensor_health[sensor] = (
            calculate_live_health(
                result
            )
        )


    alert_result = (
        generate_alert(
            inference_result
        )
    )


    return {

        "timestamp":
            inference_result.get(
                "timestamp"
            ),

        "station":
            inference_result.get(
                "station"
            ),

        "status":
            inference_result.get(
                "status"
            ),

        "primary_diagnosis":
            inference_result.get(
                "primary_diagnosis"
            ),

        "sensors":
            inference_result.get(
                "sensors"
            ),

        "sensor_health":
            sensor_health,

        "alerts":
            alert_result,

    }