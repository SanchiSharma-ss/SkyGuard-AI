from app.services.alert_engine import build_live_result

from app.services.inference_engine import run_inference

from app.services.replay_engine import replay_engine

from typing import Optional

from fastapi import FastAPI, HTTPException, Query

from app.services.data_service import (
    get_stations,
    get_latest_station_data,
    get_station_sensors,
    get_sensor_health,
    get_station_health,
    get_anomalies,
    get_statistics,
)


# ============================================================
# ATMOS FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="ATMOS API",
    description=(
        "AI/ML-Based Intelligent Anomaly Detection "
        "and Sensor Health Monitoring for "
        "Automatic Weather Stations"
    ),
    version="1.0.0",
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "ATMOS",
        "description": (
            "Automatic Weather Station "
            "Intelligent Monitoring System"
        ),
        "status": "online",
        "version": "1.0.0",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def api_health():

    return {
        "status": "healthy",
        "service": "ATMOS FastAPI Backend",
    }


# ============================================================
# STATISTICS
# ============================================================

@app.get("/statistics")
def statistics():

    return get_statistics()


# ============================================================
# STATIONS
# ============================================================

@app.get("/stations")
def stations():

    station_list = get_stations()

    return {
        "count": len(station_list),
        "stations": station_list,
    }


# ============================================================
# STATION LATEST DATA
# ============================================================

@app.get("/stations/{station}")
def station(station: str):

    data = get_latest_station_data(station)

    if data is None:

        raise HTTPException(
            status_code=404,
            detail=f"Station '{station}' not found",
        )

    return data


# ============================================================
# STATION SENSOR HEALTH
# ============================================================

@app.get("/stations/{station}/sensors")
def station_sensors(station: str):

    data = get_station_sensors(station)

    if not data:

        raise HTTPException(
            status_code=404,
            detail=(
                f"No sensor health data "
                f"found for '{station}'"
            ),
        )

    return {
        "station": station,
        "sensors": data,
    }


# ============================================================
# SENSOR HEALTH
# ============================================================

@app.get("/sensor-health")
def sensor_health(

    station: Optional[str] = None,

    sensor: Optional[str] = Query(
        default=None,
        description="T2M, RH2M or PS",
    ),

):

    data = get_sensor_health(
        station,
        sensor,
    )

    return {
        "count": len(data),
        "data": data,
    }


# ============================================================
# STATION HEALTH
# ============================================================

@app.get("/station-health")
def station_health(

    station: Optional[str] = None,

):

    data = get_station_health(
        station
    )

    return {
        "count": len(data),
        "data": data,
    }


# ============================================================
# ANOMALIES
# ============================================================

@app.get("/anomalies")
def anomalies(

    station: Optional[str] = None,

    sensor: Optional[str] = Query(
        default=None,
        description="T2M, RH2M or PS",
    ),

    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
    ),

):

    data = get_anomalies(
        station=station,
        sensor=sensor,
        limit=limit,
    )

    return {
        "count": len(data),
        "data": data,
    }


# ============================================================
# SYSTEM INFORMATION
# ============================================================

@app.get("/system-info")
def system_info():

    return {

        "name": "ATMOS",

        "version": "1.0.0",

        "backend": "FastAPI",

        "ml": [
            "Z-Score",
            "IQR",
            "Isolation Forest",
            "Autoencoder",
            "Random Forest Fault Classification",
        ],

        "sensors": [
            "T2M",
            "RH2M",
            "PS",
        ],

        "health_scoring": "0-100",

        "status": "prototype",

    }

# ============================================================
# REAL-TIME REPLAY ENGINE
# ============================================================

@app.get("/replay/status")
def replay_status():

    return replay_engine.status()


@app.post("/replay/start")
def replay_start(
    interval: float = Query(
        default=2.0,
        ge=0.1,
        le=60.0,
        description="Seconds between AWS observations"
    )
):

    return replay_engine.start(
        interval=interval
    )


@app.post("/replay/stop")
def replay_stop():

    return replay_engine.stop()


@app.post("/replay/reset")
def replay_reset():

    return replay_engine.reset()


@app.get("/replay/current")
def replay_current():

    record = (
        replay_engine
        .get_current_record()
    )

    if record is None:

        return {
            "status": "no_record"
        }

    return {
        "status": "available",
        "record": record,
    }


@app.post("/replay/next")
def replay_next():

    record = (
        replay_engine
        .process_next()
    )

    return {
        "status": "processed",
        "record": record,
    }
# ============================================================
# LIVE ML INFERENCE
# ============================================================

@app.get("/replay/inference")
def replay_inference():

    record = (
        replay_engine
        .get_current_record()
    )

    if record is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "No replay observation "
                "is currently available. "
                "Start the replay first."
            )
        )

    try:

        result = run_inference(
            record
        )

        return result

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
    # ============================================================
# COMPLETE LIVE ATMOS RESULT
# ============================================================

@app.get("/replay/live-result")
def replay_live_result():

    record = (
        replay_engine
        .get_current_record()
    )

    if record is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "No replay observation "
                "is currently available."
            )
        )


    try:

        inference = run_inference(
            record
        )

        result = build_live_result(
            inference
        )

        return result


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )