import time

import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# SkyGuard COMMAND CENTER 2.0
# ============================================================

API_BASE_URL = "http://127.0.0.1:8000"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SkyGuard AI | Weather Intelligence",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# HTML RENDER HELPER
# ============================================================

def render_html(content):
    """
    Render custom HTML using Streamlit's native HTML renderer.
    This prevents HTML tags from appearing as plain text.
    """
    st.html(content)


# ============================================================
# PREMIUM CSS
# ============================================================

render_html(
    """
<style>

:root {
    --bg: #080b12;
    --panel: #111722;
    --panel2: #151c29;
    --border: rgba(255,255,255,0.07);
    --text: #f8fafc;
    --muted: #7d899e;
    --muted2: #566176;
    --blue: #6366f1;
    --cyan: #06b6d4;
    --green: #22c55e;
    --yellow: #f59e0b;
    --red: #ef4444;
}


/* =========================================================
   APP
   ========================================================= */

.stApp {
    background:
        radial-gradient(
            circle at 10% 5%,
            rgba(79,70,229,0.12),
            transparent 25%
        ),
        radial-gradient(
            circle at 90% 10%,
            rgba(6,182,212,0.08),
            transparent 24%
        ),
        #080b12;
}

.block-container {
    max-width: 1650px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #0c1019 0%,
            #080b12 100%
        );

    border-right:
        1px solid rgba(255,255,255,0.07);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}


/* =========================================================
   HEADER
   ========================================================= */

.atmos-header {
    display: flex;
    justify-content: space-between;
    align-items: center;

    padding: 5px 0 20px 0;
}

.atmos-brand {
    display: flex;
    align-items: center;
    gap: 14px;
}

.atmos-logo {
    width: 50px;
    height: 50px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 15px;

    background:
        linear-gradient(
            135deg,
            #4f46e5,
            #06b6d4
        );

    box-shadow:
        0 0 35px rgba(79,70,229,0.25);

    font-size: 25px;
}

.atmos-title {
    font-size: 30px;
    font-weight: 850;
    letter-spacing: -1px;
    color: #f8fafc;
}

.atmos-subtitle {
    margin-top: 2px;
    color: #778399;
    font-size: 11px;
}

.system-online {
    display: flex;
    align-items: center;
    gap: 8px;

    padding: 9px 15px;

    border-radius: 999px;

    background:
        rgba(34,197,94,0.08);

    border:
        1px solid rgba(34,197,94,0.20);

    color: #86efac;

    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.5px;
}

.online-dot {
    width: 8px;
    height: 8px;

    border-radius: 50%;

    background: #22c55e;

    box-shadow:
        0 0 12px #22c55e;
}


/* =========================================================
   NAVIGATION
   ========================================================= */

.nav-label {
    color: #59657a;

    font-size: 9px;

    font-weight: 900;

    letter-spacing: 1.5px;

    margin-top: 15px;
    margin-bottom: 7px;
}


/* =========================================================
   KPI CARDS
   ========================================================= */

.kpi-card {
    position: relative;

    padding: 20px;

    min-height: 125px;

    border-radius: 17px;

    background:
        linear-gradient(
            145deg,
            rgba(22,28,42,0.98),
            rgba(12,16,25,0.98)
        );

    border:
        1px solid rgba(255,255,255,0.075);

    box-shadow:
        0 15px 40px rgba(0,0,0,0.20);

    overflow: hidden;
}

.kpi-card::after {
    content: "";

    position: absolute;

    width: 110px;
    height: 110px;

    right: -55px;
    top: -55px;

    border-radius: 50%;

    background:
        rgba(99,102,241,0.10);
}

.kpi-label {
    color: #7f8ba1;

    font-size: 9px;

    font-weight: 800;

    letter-spacing: 1px;
}

.kpi-value {
    color: #f8fafc;

    font-size: 31px;

    font-weight: 850;

    margin-top: 8px;
}

.kpi-caption {
    color: #657186;

    font-size: 10px;

    margin-top: 5px;
}


/* =========================================================
   SECTION
   ========================================================= */

.section-heading {
    display: flex;

    justify-content: space-between;

    align-items: center;

    margin-top: 25px;
    margin-bottom: 12px;
}

.section-title {
    color: #f1f5f9;

    font-size: 18px;

    font-weight: 800;
}

.section-description {
    color: #657186;

    font-size: 10px;

    margin-top: 3px;
}


/* =========================================================
   PANEL
   ========================================================= */

.panel {
    background:
        linear-gradient(
            145deg,
            rgba(20,26,39,0.97),
            rgba(11,15,24,0.97)
        );

    border:
        1px solid rgba(255,255,255,0.065);

    border-radius: 17px;

    padding: 19px;

    box-shadow:
        0 14px 40px rgba(0,0,0,0.17);
}


/* =========================================================
   LIVE BANNER
   ========================================================= */

.live-banner {
    display: flex;

    justify-content: space-between;

    align-items: center;

    padding: 18px 22px;

    border-radius: 17px;

    margin-bottom: 17px;

    background:
        linear-gradient(
            100deg,
            rgba(34,197,94,0.08),
            rgba(6,182,212,0.05)
        );

    border:
        1px solid rgba(34,197,94,0.16);
}

.live-banner.anomaly {
    background:
        linear-gradient(
            100deg,
            rgba(239,68,68,0.13),
            rgba(245,158,11,0.05)
        );

    border:
        1px solid rgba(239,68,68,0.22);
}

.live-title {
    font-size: 21px;
    font-weight: 850;
}

.live-meta {
    color: #748198;
    font-size: 10px;
    margin-top: 4px;
}


/* =========================================================
   SENSOR CARDS
   ========================================================= */

.sensor-card {
    padding: 19px;

    border-radius: 17px;

    background:
        linear-gradient(
            145deg,
            #131a27,
            #0d121c
        );

    border:
        1px solid rgba(255,255,255,0.07);

    min-height: 270px;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.14);
}

.sensor-name {
    color: #93a0b5;

    font-size: 11px;

    font-weight: 900;

    letter-spacing: 1.2px;
}

.sensor-value {
    color: #f8fafc;

    font-size: 30px;

    font-weight: 850;

    margin-top: 10px;
}

.sensor-status {
    margin-top: 8px;

    font-size: 11px;

    font-weight: 900;
}

.sensor-stat {
    display: flex;

    justify-content: space-between;

    gap: 10px;

    padding: 7px 0;

    border-bottom:
        1px solid rgba(255,255,255,0.045);

    font-size: 10px;
}

.sensor-stat-label {
    color: #657186;
}

.sensor-stat-value {
    color: #dbe4f0;

    font-weight: 750;
}


/* =========================================================
   ALERTS
   ========================================================= */

.alert-card {
    padding: 15px 17px;

    border-radius: 14px;

    margin-bottom: 9px;

    background:
        rgba(239,68,68,0.055);

    border:
        1px solid rgba(239,68,68,0.18);
}

.alert-card.high {
    background:
        rgba(245,158,11,0.055);

    border:
        1px solid rgba(245,158,11,0.18);
}

.alert-header {
    display: flex;

    justify-content: space-between;

    align-items: center;
}

.alert-title {
    color: #f8fafc;

    font-size: 12px;

    font-weight: 850;
}

.alert-severity {
    font-size: 9px;

    font-weight: 900;

    letter-spacing: 0.8px;
}


/* =========================================================
   HEALTH
   ========================================================= */

.health-card {
    padding: 18px;

    border-radius: 15px;

    background:
        rgba(255,255,255,0.025);

    border:
        1px solid rgba(255,255,255,0.055);
}

.health-score {
    font-size: 29px;

    font-weight: 850;

    margin-top: 5px;
}


/* =========================================================
   PIPELINE
   ========================================================= */

.pipeline {
    display: flex;

    align-items: center;

    justify-content: space-between;

    gap: 7px;

    padding: 15px 2px;
}

.pipeline-node {
    flex: 1;

    text-align: center;

    padding: 15px 7px;

    border-radius: 12px;

    background:
        rgba(255,255,255,0.035);

    border:
        1px solid rgba(255,255,255,0.06);
}

.pipeline-icon {
    font-size: 20px;
}

.pipeline-name {
    margin-top: 6px;

    font-size: 9px;

    font-weight: 850;

    color: #dbe4f0;
}

.pipeline-arrow {
    color: #455166;

    font-size: 17px;
}


/* =========================================================
   FOOTER
   ========================================================= */

.atmos-footer {
    margin-top: 35px;

    padding-top: 15px;

    border-top:
        1px solid rgba(255,255,255,0.05);

    color: #465267;

    font-size: 9px;

    text-align: center;
}


/* =========================================================
   STREAMLIT
   ========================================================= */

.stButton button {
    border-radius: 10px;

    border:
        1px solid rgba(255,255,255,0.08);

    background:
        rgba(255,255,255,0.035);

    color: #dbe4f0;

    font-weight: 700;
}

.stButton button:hover {
    border-color:
        rgba(99,102,241,0.55);

    color: white;
}

</style>
"""
)


# ============================================================
# API FUNCTIONS
# ============================================================

def api_get(endpoint, timeout=5):

    try:

        response = requests.get(
            f"{API_BASE_URL}{endpoint}",
            timeout=timeout,
        )

        response.raise_for_status()

        return response.json()

    except Exception as exc:

        return {
            "_error": str(exc)
        }


def api_post(endpoint, payload=None, timeout=5):

    try:

        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=payload,
            timeout=timeout,
        )

        response.raise_for_status()

        return response.json()

    except Exception as exc:

        return {
            "_error": str(exc)
        }


# ============================================================
# HELPERS
# ============================================================

def number(value, default=0.0):

    try:

        if value is None:
            return default

        value = float(value)

        if pd.isna(value):
            return default

        return value

    except Exception:

        return default


def clean_station(value):

    value = str(value)

    if value.startswith("(") and value.endswith(",)"):

        value = value[1:-2]

    return value.strip(
        "'\" "
    )


def status_icon(status):

    status = str(
        status
    ).upper()

    if status == "ANOMALY":
        return "🔴"

    if status in [
        "WARNING",
        "DEGRADED",
    ]:
        return "🟡"

    if status in [
        "NORMAL",
        "HEALTHY",
        "EXCELLENT",
    ]:
        return "🟢"

    return "⚪"


def severity_icon(severity):

    severity = str(
        severity
    ).upper()

    if severity == "CRITICAL":
        return "🔴"

    if severity == "HIGH":
        return "🟠"

    if severity == "MEDIUM":
        return "🟡"

    if severity == "LOW":
        return "🔵"

    return "🟢"


# ============================================================
# HEADER
# ============================================================

render_html(
    """
<div class="atmos-header">

    <div class="atmos-brand">

        <div class="atmos-logo">
            🌦️
        </div>

        <div>

            <div class="atmos-title">
                SkyGuard AI
            </div>

            <div class="atmos-subtitle">
                Intelligent Weather Station Monitoring Platform
            </div>

        </div>

    </div>

    <div class="system-online">
        <span class="online-dot"></span>
        SkyGuard CORE
    </div>

</div>
"""
)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:

    st.session_state.page = "Overview"


if "auto_refresh" not in st.session_state:

    st.session_state.auto_refresh = True


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html(
        """
        <div style="
            font-size:20px;
            font-weight:850;
            color:#f8fafc;
        ">
            SkyGuard
        </div>

        <div style="
            font-size:9px;
            color:#5d6a80;
            letter-spacing:1px;
            margin-top:3px;
        ">
            WEATHER INTELLIGENCE
        </div>
        """
    )


    render_html(
        """
        <div class="nav-label">
            COMMAND CENTER
        </div>
        """
    )


    pages = [
        ("⌂", "Overview"),
        ("◉", "Live Monitor"),
        ("⚠", "Anomaly Explorer"),
        ("♥", "Sensor Health"),
        ("◎", "Station Network"),
        ("▣", "Data Explorer"),
        ("◈", "How SkyGuard Works"),
        ("⚙", "System Status"),
    ]


    for icon, page_name in pages:

        if st.button(
            f"{icon}  {page_name}",
            key=f"nav_{page_name}",
            use_container_width=True,
        ):

            st.session_state.page = page_name

            st.rerun()


    st.divider()


    render_html(
        """
        <div class="nav-label">
            LIVE REPLAY
        </div>
        """
    )


    replay_status = api_get(
        "/replay/status"
    )


    if "_error" in replay_status:

        st.error(
            "FastAPI Offline"
        )

    else:

        running = replay_status.get(
            "running",
            False
        )


        if running:

            st.success(
                "● Replay Running"
            )

        else:

            st.warning(
                "● Replay Stopped"
            )


        interval = st.slider(
            "Replay speed",
            min_value=0.1,
            max_value=5.0,
            value=0.5,
            step=0.1,
        )


        c1, c2 = st.columns(2)


        with c1:

            if st.button(
                "▶ Start",
                key="start_replay",
                use_container_width=True,
            ):

                result = api_post(
                    "/replay/start",
                    {
                        "interval_seconds": interval
                    },
                )

                if "_error" not in result:

                    st.rerun()

                else:

                    st.error(
                        result["_error"]
                    )


        with c2:

            if st.button(
                "■ Stop",
                key="stop_replay",
                use_container_width=True,
            ):

                result = api_post(
                    "/replay/stop"
                )

                if "_error" not in result:

                    st.rerun()


        if st.button(
            "↻ Reset Replay",
            key="reset_replay",
            use_container_width=True,
        ):

            api_post(
                "/replay/reset"
            )

            st.rerun()


    st.divider()


    render_html(
        """
        <div class="nav-label">
            DISPLAY
        </div>
        """
    )


    st.session_state.auto_refresh = st.checkbox(
        "Auto refresh",
        value=st.session_state.auto_refresh,
    )


    refresh_rate = st.slider(
        "Refresh interval",
        min_value=1,
        max_value=10,
        value=2,
    )


    render_html(
        """
        <div style="
            margin-top:18px;
            padding:12px;
            border-radius:12px;
            background:rgba(255,255,255,0.025);
            border:1px solid rgba(255,255,255,0.05);
        ">

            <div style="
                font-size:9px;
                color:#59657a;
                font-weight:900;
                letter-spacing:1px;
            ">
                DEMO MODE
            </div>

            <div style="
                font-size:10px;
                color:#78859a;
                margin-top:5px;
                line-height:1.55;
            ">
                Historical AWS observations are
                replayed as a real-time monitoring
                stream.
            </div>

        </div>
        """
    )


# ============================================================
# FETCH DATA
# ============================================================

live_result = api_get(
    "/replay/live-result"
)

current_observation = api_get(
    "/replay/current"
)

statistics = api_get(
    "/statistics"
)

stations_data = api_get(
    "/stations"
)


# ============================================================
# COMBINE CURRENT RAW OBSERVATION
# ============================================================

if (
    "_error" not in current_observation
    and isinstance(current_observation, dict)
):

    if "T2M" in current_observation:

        live_result["T2M"] = current_observation["T2M"]

    if "RH2M" in current_observation:

        live_result["RH2M"] = current_observation["RH2M"]

    if "PS" in current_observation:

        live_result["PS"] = current_observation["PS"]

    if "timestamp" not in live_result:

        live_result["timestamp"] = current_observation.get(
            "timestamp"
        )

    if "station" not in live_result:

        live_result["station"] = current_observation.get(
            "station"
        )


# ============================================================
# CORE CONNECTION CHECK
# ============================================================

FASTAPI_ONLINE = (
    "_error" not in live_result
    or "_error" not in replay_status
)


# ============================================================
# OVERVIEW
# ============================================================

def render_overview():

    render_html(
        """
        <div class="section-heading">

            <div>

                <div class="section-title">
                    Network Overview
                </div>

                <div class="section-description">
                    Real-time operational status of the SkyGuard
                    weather station network
                </div>

            </div>

        </div>
        """
    )


    # --------------------------------------------------------
    # STATIONS
    # --------------------------------------------------------

    station_count = 41


    if isinstance(stations_data, list):

        station_count = len(
            stations_data
        )

    elif isinstance(stations_data, dict):

        possible = (
            stations_data.get("stations")
            or stations_data.get("data")
            or []
        )

        if isinstance(possible, list):

            station_count = len(
                possible
            )


    sensor_count = (
        station_count * 3
    )


    alert_count = 0

    system_status = "NORMAL"


    if (
        "_error" not in live_result
        and "detail" not in live_result
    ):

        alerts = live_result.get(
            "alerts",
            {}
        )

        alert_count = alerts.get(
            "alert_count",
            0
        )

        system_status = live_result.get(
            "status",
            "NORMAL"
        )


    health_values = []


    if (
        "_error" not in live_result
        and "detail" not in live_result
    ):

        for info in live_result.get(
            "sensor_health",
            {}
        ).values():

            health_values.append(
                number(
                    info.get(
                        "score",
                        0
                    )
                )
            )


    average_health = (
        sum(health_values)
        / len(health_values)
        if health_values
        else 0
    )


    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    columns = st.columns(4)


    kpis = [
        (
            "MONITORED STATIONS",
            station_count,
            "AWS stations in network",
        ),
        (
            "SENSOR STREAMS",
            sensor_count,
            "T2M • RH2M • PS",
        ),
        (
            "ACTIVE ALERTS",
            alert_count,
            "Current observation",
        ),
        (
            "AVERAGE HEALTH",
            (
                f"{average_health:.0f}/100"
                if health_values
                else "—"
            ),
            "Live sensor condition",
        ),
    ]


    for col, (
        label,
        value,
        caption,
    ) in zip(
        columns,
        kpis,
    ):

        with col:

            render_html(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        {label}
                    </div>

                    <div class="kpi-value">
                        {value}
                    </div>

                    <div class="kpi-caption">
                        {caption}
                    </div>

                </div>
                """
            )


    # --------------------------------------------------------
    # BACKEND OFFLINE
    # --------------------------------------------------------

    if (
        "_error" in live_result
        or "detail" in live_result
    ):

        render_html(
            """
            <div class="panel"
                 style="
                 margin-top:20px;
                 text-align:center;
                 padding:55px 25px;
                 ">

                <div style="
                    font-size:42px;
                ">
                    ⚡
                </div>

                <div style="
                    font-size:21px;
                    font-weight:850;
                    margin-top:12px;
                ">
                    SkyGuard Core Offline
                </div>

                <div style="
                    color:#68758a;
                    font-size:11px;
                    margin-top:7px;
                ">
                    FastAPI is not currently connected.
                </div>

                <div style="
                    margin-top:15px;
                    padding:10px;
                    display:inline-block;
                    border-radius:9px;
                    background:rgba(239,68,68,0.08);
                    border:1px solid rgba(239,68,68,0.15);
                    color:#fca5a5;
                    font-size:10px;
                    font-weight:700;
                ">
                    API: http://127.0.0.1:8000
                </div>

                <div style="
                    color:#59657a;
                    font-size:10px;
                    margin-top:15px;
                ">
                    Start FastAPI and refresh this dashboard.
                </div>

            </div>
            """
        )

        return


    # --------------------------------------------------------
    # CURRENT SIGNAL
    # --------------------------------------------------------

    station = clean_station(
        live_result.get(
            "station",
            "-"
        )
    )


    status = live_result.get(
        "status",
        "NORMAL"
    )


    timestamp = live_result.get(
        "timestamp",
        "-"
    )


    banner_class = (
        "anomaly"
        if status == "ANOMALY"
        else ""
    )


    render_html(
        f"""
        <div class="section-heading">

            <div class="section-title">
                Current Network Signal
            </div>

        </div>

        <div class="live-banner {banner_class}">

            <div>

                <div class="live-title">
                    {status_icon(status)}
                    {station.upper()}
                </div>

                <div class="live-meta">
                    Latest observation:
                    {timestamp}
                </div>

            </div>

            <div style="
                font-size:12px;
                font-weight:900;
            ">
                {status}
            </div>

        </div>
        """
    )


    # --------------------------------------------------------
    # SENSOR SNAPSHOT
    # --------------------------------------------------------

    render_sensor_cards(
        live_result
    )


    # --------------------------------------------------------
    # LOWER SECTION
    # --------------------------------------------------------

    left, right = st.columns(
        [1.2, 0.8]
    )


    with left:

        render_html(
            """
            <div class="section-heading">

                <div class="section-title">
                    Active Alerts
                </div>

            </div>
            """
        )

        render_alerts(
            live_result
        )


    with right:

        render_html(
            """
            <div class="section-heading">

                <div class="section-title">
                    Primary Diagnosis
                </div>

            </div>
            """
        )

        render_primary_diagnosis(
            live_result
        )


# ============================================================
# SENSOR CARDS
# ============================================================

def render_sensor_cards(data):

    sensors = data.get(
        "sensors",
        {}
    )


    health = data.get(
        "sensor_health",
        {}
    )


    columns = st.columns(3)


    sensor_units = {
        "T2M": "°C",
        "RH2M": "%",
        "PS": "",
    }


    for col, sensor_name in zip(
        columns,
        ["T2M", "RH2M", "PS"],
    ):

        sensor = sensors.get(
            sensor_name,
            {}
        )


        health_info = health.get(
            sensor_name,
            {}
        )


        fault = sensor.get(
            "fault",
            "normal"
        )


        confidence = number(
            sensor.get(
                "confidence",
                0
            )
        )


        local_z = number(
            sensor.get(
                "local_z",
                0
            )
        )


        anomaly = bool(
            sensor.get(
                "anomaly",
                False
            )
        )


        severity = sensor.get(
            "severity",
            "NORMAL"
        )


        health_score = number(
            health_info.get(
                "score",
                100
            )
        )


        raw_value = data.get(
            sensor_name
        )


        if raw_value is None:

            raw_value = (
                current_observation.get(
                    sensor_name
                )
                if isinstance(
                    current_observation,
                    dict
                )
                else None
            )


        unit = sensor_units.get(
            sensor_name,
            ""
        )


        if raw_value is not None:

            value_text = (
                f"{number(raw_value):.2f}"
                f"{unit}"
            )

        else:

            value_text = "—"


        border = (
            "rgba(239,68,68,0.30)"
            if anomaly
            else "rgba(255,255,255,0.07)"
        )


        with col:

            render_html(
                f"""
                <div class="sensor-card"
                     style="
                     border-color:{border};
                     ">

                    <div class="sensor-name">
                        {sensor_name}
                    </div>

                    <div class="sensor-value">
                        {value_text}
                    </div>

                    <div class="sensor-status">
                        {severity_icon(severity)}
                        {severity}
                    </div>

                    <div style="
                        margin-top:13px;
                    ">

                        <div class="sensor-stat">

                            <span class="sensor-stat-label">
                                Classifier
                            </span>

                            <span class="sensor-stat-value">
                                {str(fault).upper()}
                            </span>

                        </div>

                        <div class="sensor-stat">

                            <span class="sensor-stat-label">
                                Confidence
                            </span>

                            <span class="sensor-stat-value">
                                {confidence:.2f}%
                            </span>

                        </div>

                        <div class="sensor-stat">

                            <span class="sensor-stat-label">
                                Local Z-score
                            </span>

                            <span class="sensor-stat-value">
                                {local_z:+.2f}
                            </span>

                        </div>

                        <div class="sensor-stat">

                            <span class="sensor-stat-label">
                                Anomaly
                            </span>

                            <span class="sensor-stat-value">
                                {"YES" if anomaly else "NO"}
                            </span>

                        </div>

                        <div class="sensor-stat">

                            <span class="sensor-stat-label">
                                Sensor Health
                            </span>

                            <span class="sensor-stat-value">
                                {health_score:.0f}/100
                            </span>

                        </div>

                    </div>

                </div>
                """
            )


# ============================================================
# PRIMARY DIAGNOSIS
# ============================================================

def render_primary_diagnosis(data):

    primary = data.get(
        "primary_diagnosis",
        {}
    )


    sensor = primary.get(
        "sensor"
    )


    fault = primary.get(
        "fault",
        "normal"
    )


    confidence = number(
        primary.get(
            "confidence",
            0
        )
    )


    severity = primary.get(
        "severity",
        "NORMAL"
    )


    if not sensor:

        render_html(
            """
            <div class="panel"
                 style="text-align:center;">

                <div style="
                    font-size:35px;
                    margin-top:5px;
                ">
                    🟢
                </div>

                <div style="
                    font-size:16px;
                    font-weight:850;
                    margin-top:6px;
                ">
                    No Primary Anomaly
                </div>

                <div style="
                    color:#68758a;
                    font-size:10px;
                    margin-top:5px;
                    line-height:1.5;
                ">
                    Current observation is within
                    active detection criteria.
                </div>

            </div>
            """
        )

        return


    render_html(
        f"""
        <div class="panel"
             style="
             border-color:
             rgba(239,68,68,0.25);
             ">

            <div style="
                color:#8491a7;
                font-size:9px;
                font-weight:900;
                letter-spacing:1px;
            ">
                PRIMARY SENSOR
            </div>

            <div style="
                font-size:29px;
                font-weight:850;
                margin-top:6px;
            ">
                {sensor}
            </div>

            <div style="
                color:#fca5a5;
                font-size:12px;
                font-weight:850;
                margin-top:5px;
            ">
                {severity_icon(severity)}
                {severity}
            </div>

            <hr style="
                border:none;
                border-top:
                1px solid rgba(255,255,255,0.06);
                margin:14px 0;
            ">

            <div class="sensor-stat">

                <span class="sensor-stat-label">
                    Diagnosis
                </span>

                <span class="sensor-stat-value">
                    {fault}
                </span>

            </div>

            <div class="sensor-stat">

                <span class="sensor-stat-label">
                    Confidence
                </span>

                <span class="sensor-stat-value">
                    {confidence:.2f}%
                </span>

            </div>

        </div>
        """
    )


# ============================================================
# ALERTS
# ============================================================

def render_alerts(data):

    alerts_data = data.get(
        "alerts",
        {}
    )


    alerts = alerts_data.get(
        "alerts",
        []
    )


    if not alerts:

        render_html(
            """
            <div class="panel">

                <div style="
                    font-size:24px;
                ">
                    🟢
                </div>

                <div style="
                    font-size:14px;
                    font-weight:850;
                    margin-top:5px;
                ">
                    No active alerts
                </div>

                <div style="
                    color:#68758a;
                    font-size:10px;
                    margin-top:4px;
                ">
                    SkyGuard is not currently reporting
                    an active sensor anomaly.
                </div>

            </div>
            """
        )

        return


    for alert in alerts:

        severity = alert.get(
            "severity",
            "HIGH"
        )


        card_class = (
            "high"
            if severity == "HIGH"
            else ""
        )


        render_html(
            f"""
            <div class="alert-card {card_class}">

                <div class="alert-header">

                    <div class="alert-title">
                        {severity_icon(severity)}
                        {alert.get('sensor', 'UNKNOWN')}
                        anomaly
                    </div>

                    <div class="alert-severity">
                        {severity}
                    </div>

                </div>

                <div style="
                    margin-top:8px;
                    color:#8793a7;
                    font-size:10px;
                ">

                    Confidence:
                    <b>
                    {number(alert.get('confidence',0)):.2f}%
                    </b>

                    &nbsp; | &nbsp;

                    Health:
                    <b>
                    {number(alert.get('sensor_health',0)):.0f}/100
                    </b>

                    &nbsp; | &nbsp;

                    Z:
                    <b>
                    {number(alert.get('local_z',0)):+.2f}
                    </b>

                </div>

            </div>
            """
        )


        evidence = alert.get(
            "evidence",
            []
        )


        if evidence:

            with st.expander(
                f"Evidence — {alert.get('sensor', 'Sensor')}"
            ):

                for item in evidence:

                    st.write(
                        f"• {item}"
                    )


# ============================================================
# LIVE MONITOR
# ============================================================

def render_live_monitor():

    render_html(
        """
        <div class="section-heading">

            <div>

                <div class="section-title">
                    Live Monitoring
                </div>

                <div class="section-description">
                    Current observation processed through
                    the SkyGuard inference pipeline
                </div>

            </div>

        </div>
        """
    )

    if "_error" in live_result:

        render_offline()

        return


    if "detail" in live_result:

        st.warning(
            live_result["detail"]
        )

        return


    station = clean_station(
        live_result.get(
            "station",
            "-"
        )
    )


    status = live_result.get(
        "status",
        "NORMAL"
    )


    timestamp = live_result.get(
        "timestamp",
        "-"
    )


    banner_class = (
        "anomaly"
        if status == "ANOMALY"
        else ""
    )


    render_html(
        f"""
        <div class="live-banner {banner_class}">

            <div>

                <div class="live-title">
                    {status_icon(status)}
                    {station.upper()}
                </div>

                <div class="live-meta">
                    {timestamp}
                </div>

            </div>

            <div style="
                font-size:12px;
                font-weight:900;
            ">
                {status}
            </div>

        </div>
        """
    )


    render_sensor_cards(
        live_result
    )


    render_html(
        """
        <div class="section-heading">

            <div class="section-title">
                Detection Evidence
            </div>

        </div>
        """
    )


    primary = live_result.get(
        "primary_diagnosis",
        {}
    )


    if primary.get("sensor"):

        sensor = primary.get(
            "sensor"
        )


        sensor_data = live_result.get(
            "sensors",
            {}
        ).get(
            sensor,
            {}
        )


        z = number(
            sensor_data.get(
                "local_z",
                0
            )
        )


        render_html(
            f"""
            <div class="panel">

                <div style="
                    font-size:15px;
                    font-weight:850;
                ">
                    Why did SkyGuard trigger this event?
                </div>

                <div style="
                    margin-top:13px;
                    color:#8c99ad;
                    font-size:11px;
                    line-height:1.9;
                ">

                    ✓ Sensor:
                    <b>{sensor}</b><br>

                    ✓ Detection:
                    <b>{primary.get('fault')}</b><br>

                    ✓ Local Z-score:
                    <b>{z:+.2f}</b><br>

                    ✓ Detection threshold:
                    <b>|z| ≥ 3</b><br>

                    ✓ Severity:
                    <b>{primary.get('severity')}</b><br>

                    ✓ Confidence:
                    <b>
                    {number(primary.get('confidence',0)):.2f}%
                    </b>

                </div>

            </div>
            """
        )

    else:

        st.success(
            "No primary anomaly detected."
        )


# ============================================================
# ANOMALY EXPLORER
# ============================================================

def render_anomaly_explorer():

    render_html(
        """
        <div class="section-heading">

            <div>

                <div class="section-title">
                    Anomaly Explorer
                </div>

                <div class="section-description">
                    Detailed sensor-level anomaly evidence
                </div>

            </div>

        </div>
        """
    )


    if (
        "_error" in live_result
        or "detail" in live_result
    ):

        render_offline()

        return


    rows = []


    for sensor_name, item in live_result.get(
        "sensors",
        {}
    ).items():

        rows.append(
            {
                "Sensor": sensor_name,

                "Fault Classifier": str(
                    item.get(
                        "fault",
                        "normal"
                    )
                ).upper(),

                "Confidence %": number(
                    item.get(
                        "confidence",
                        0
                    )
                ),

                "Local Z": number(
                    item.get(
                        "local_z",
                        0
                    )
                ),

                "Anomaly": (
                    "YES"
                    if item.get(
                        "anomaly",
                        False
                    )
                    else "NO"
                ),

                "Severity": item.get(
                    "severity",
                    "NORMAL"
                ),
            }
        )


    df = pd.DataFrame(
        rows
    )


    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


    if not df.empty:

        fig = go.Figure()


        fig.add_trace(
            go.Bar(
                x=df["Sensor"],
                y=df["Local Z"],
                text=[
                    f"{x:+.2f}"
                    for x in df["Local Z"]
                ],
                textposition="auto",
            )
        )


        fig.add_hline(
            y=3,
            line_dash="dash",
            annotation_text="Upper threshold",
        )


        fig.add_hline(
            y=-3,
            line_dash="dash",
            annotation_text="Lower threshold",
        )


        fig.update_layout(
            title="Local Z-Score Analysis",
            height=430,
            template="plotly_dark",
            yaxis_title="Local Z-Score",
            xaxis_title="Sensor",
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# SENSOR HEALTH
# ============================================================

def render_sensor_health():

    render_html(
        """
        <div class="section-heading">

            <div>

                <div class="section-title">
                    Sensor Health Intelligence
                </div>

                <div class="section-description">
                    Current operational reliability of each sensor
                </div>

            </div>

        </div>
        """
    )


    if (
        "_error" in live_result
        or "detail" in live_result
    ):

        render_offline()

        return


    health = live_result.get(
        "sensor_health",
        {}
    )


    columns = st.columns(
        len(health)
        if health
        else 1
    )


    rows = []


    for col, (
        sensor,
        info,
    ) in zip(
        columns,
        health.items(),
    ):

        score = number(
            info.get(
                "score",
                0
            )
        )


        status = info.get(
            "status",
            "UNKNOWN"
        )


        rows.append(
            {
                "Sensor": sensor,
                "Health": score,
            }
        )


        with col:

            render_html(
                f"""
                <div class="health-card">

                    <div style="
                        color:#8b98ac;
                        font-size:9px;
                        font-weight:900;
                        letter-spacing:1px;
                    ">
                        {sensor}
                    </div>

                    <div class="health-score">
                        {score:.0f}
                        <span style="
                            color:#667286;
                            font-size:11px;
                        ">
                            /100
                        </span>
                    </div>

                    <div style="
                        color:#94a3b8;
                        font-size:10px;
                        font-weight:750;
                        margin-top:3px;
                    ">
                        {status}
                    </div>

                </div>
                """
            )


    if rows:

        df = pd.DataFrame(
            rows
        )


        fig = go.Figure()


        fig.add_trace(
            go.Bar(
                x=df["Sensor"],
                y=df["Health"],
                text=[
                    f"{value:.0f}"
                    for value in df["Health"]
                ],
                textposition="auto",
            )
        )


        fig.update_layout(
            title="Current Sensor Health",
            yaxis=dict(
                range=[0, 100],
                title="Health Score",
            ),
            height=400,
            template="plotly_dark",
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    st.info(
        "The current live health score is generated by the "
        "SkyGuard health engine from anomaly evidence and sensor "
        "behaviour."
    )


# ============================================================
# STATION NETWORK
# ============================================================

def render_station_network():

    render_html(
        """
        <div class="section-heading">

            <div>

                <div class="section-title">
                    Station Network
                </div>

                <div class="section-description">
                    Automatic Weather Station network
                </div>

            </div>

        </div>
        """
    )


    station_names = []


    if isinstance(
        stations_data,
        list,
    ):

        station_names = [
            clean_station(x)
            for x in stations_data
        ]


    elif isinstance(
        stations_data,
        dict,
    ):

        possible = (
            stations_data.get(
                "stations"
            )
            or stations_data.get(
                "data"
            )
            or []
        )


        if isinstance(
            possible,
            list,
        ):

            station_names = [
                clean_station(x)
                for x in possible
            ]


    if not station_names:

        station_names = [
            "Guwahati",
            "Jowai",
            "Mangaldai",
            "Tura",
            "Shillong",
        ]


    search = st.text_input(
        "Search station",
        placeholder="Search by station name...",
    )


    if search:

        station_names = [
            station
            for station in station_names
            if search.lower()
            in station.lower()
        ]


    rows = []


    for station in station_names:

        rows.append(
            {
                "Station": station,
                "Network Status": "ACTIVE",
                "Sensors": 3,
            }
        )


    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# DATA EXPLORER
# ============================================================

def render_data_explorer():

    render_html(
        """
        <div class="section-heading">

            <div>

                <div class="section-title">
                    Data Explorer
                </div>

                <div class="section-description">
                    Current AWS observation and ML evidence
                </div>

            </div>

        </div>
        """
    )


    if (
        "_error" in live_result
        or "detail" in live_result
    ):

        render_offline()

        return


    station = clean_station(
        live_result.get(
            "station",
            "-"
        )
    )


    timestamp = live_result.get(
        "timestamp",
        "-"
    )


    raw_data = {
        "Station": station,
        "Timestamp": timestamp,
        "T2M": live_result.get(
            "T2M"
        ),
        "RH2M": live_result.get(
            "RH2M"
        ),
        "PS": live_result.get(
            "PS"
        ),
        "System Status": live_result.get(
            "status",
            "NORMAL"
        ),
    }


    st.dataframe(
        pd.DataFrame(
            [raw_data]
        ),
        use_container_width=True,
        hide_index=True,
    )


    st.markdown(
        "### Sensor Detection Evidence"
    )


    rows = []


    for sensor, item in live_result.get(
        "sensors",
        {}
    ).items():

        rows.append(
            {
                "Sensor": sensor,
                "Fault": item.get(
                    "fault"
                ),
                "Confidence": item.get(
                    "confidence"
                ),
                "Local Z": item.get(
                    "local_z"
                ),
                "Anomaly": item.get(
                    "anomaly"
                ),
                "Severity": item.get(
                    "severity"
                ),
            }
        )


    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# HOW SkyGuard WORKS
# ============================================================

def render_how_atmos_works():

    render_html(
        """
        <div class="section-heading">

            <div>

                <div class="section-title">
                    How SkyGuard Works
                </div>

                <div class="section-description">
                    End-to-end explainable monitoring architecture
                </div>

            </div>

        </div>
        """
    )


    render_html(
        """
        <div class="panel">

            <div style="
                font-size:20px;
                font-weight:850;
            ">
                SkyGuard Detection Pipeline
            </div>

            <div style="
                color:#718096;
                font-size:10px;
                margin-top:5px;
            ">
                From Automatic Weather Station observations
                to explainable operational alerts.
            </div>

        </div>
        """
    )


    pipeline = [
        ("📡", "AWS DATA", "Sensor observations"),
        ("✓", "VALIDATION", "Quality checks"),
        ("⚙", "FEATURES", "Temporal + statistical"),
        ("◉", "DETECTION", "Z-score + ML"),
        ("🧠", "CLASSIFICATION", "Fault identification"),
        ("♥", "HEALTH", "Sensor reliability"),
        ("⚠", "ALERTS", "Operational response"),
    ]


    html = '<div class="pipeline">'


    for index, (
        icon,
        name,
        description,
    ) in enumerate(
        pipeline
    ):

        html += f"""
        <div class="pipeline-node">

            <div class="pipeline-icon">
                {icon}
            </div>

            <div class="pipeline-name">
                {name}
            </div>

            <div style="
                color:#566278;
                font-size:8px;
                margin-top:4px;
            ">
                {description}
            </div>

        </div>
        """


        if index < len(pipeline) - 1:

            html += (
                '<div class="pipeline-arrow">→</div>'
            )


    html += "</div>"


    render_html(
        html
    )


    explanations = [

        (
            "01",
            "Data Ingestion",
            "SkyGuard receives Automatic Weather Station "
            "observations. In this prototype, historical "
            "observations are replayed as a live stream."
        ),

        (
            "02",
            "Data Validation",
            "The preprocessing layer checks data quality, "
            "timestamps, duplicates and missing information."
        ),

        (
            "03",
            "Feature Engineering",
            "Raw sensor values are transformed into differences, "
            "percentage changes, rolling statistics, local "
            "z-scores and cross-sensor relationships."
        ),

        (
            "04",
            "Anomaly Detection",
            "SkyGuard evaluates whether current sensor behaviour "
            "is statistically unusual and also uses machine "
            "learning models."
        ),

        (
            "05",
            "Fault Classification",
            "Random Forest classifiers estimate the fault "
            "category for temperature, humidity and pressure "
            "sensors."
        ),

        (
            "06",
            "Sensor Health",
            "Detected evidence is converted into an operational "
            "0–100 sensor-health score."
        ),

        (
            "07",
            "Explainable Alert",
            "The alert engine records the sensor, severity, "
            "confidence, statistical evidence and health impact."
        ),
    ]


    for number_id, title, description in explanations:

        render_html(
            f"""
            <div class="panel"
                 style="margin-top:10px;">

                <div style="
                    display:flex;
                    gap:15px;
                ">

                    <div style="
                        color:#6366f1;
                        font-size:11px;
                        font-weight:900;
                    ">
                        {number_id}
                    </div>

                    <div>

                        <div style="
                            font-size:14px;
                            font-weight:850;
                        ">
                            {title}
                        </div>

                        <div style="
                            color:#778398;
                            font-size:10px;
                            line-height:1.7;
                            margin-top:4px;
                        ">
                            {description}
                        </div>

                    </div>

                </div>

            </div>
            """
        )


# ============================================================
# SYSTEM STATUS
# ============================================================

def render_system_status():

    render_html(
        """
        <div class="section-heading">

            <div>

                <div class="section-title">
                    System Status
                </div>

                <div class="section-description">
                    SkyGuard backend service health
                </div>

            </div>

        </div>
        """
    )


    endpoints = [
        "/",
        "/health",
        "/statistics",
        "/stations",
        "/replay/status",
        "/replay/current",
        "/replay/inference",
        "/replay/live-result",
    ]


    rows = []


    for endpoint in endpoints:

        result = api_get(
            endpoint
        )


        rows.append(
            {
                "Endpoint": endpoint,
                "Status": (
                    "ONLINE"
                    if "_error" not in result
                    else "OFFLINE"
                ),
            }
        )


    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


    render_html(
        """
        <div class="panel"
             style="margin-top:15px;">

            <div style="
                font-size:15px;
                font-weight:850;
            ">
                Backend Architecture
            </div>

            <div style="
                color:#748197;
                font-size:10px;
                line-height:1.9;
                margin-top:10px;
            ">

                Streamlit Dashboard
                →
                FastAPI
                →
                Replay Engine
                →
                Inference Engine
                →
                Fault Classifiers
                →
                Sensor Health
                →
                Alert Engine

            </div>

        </div>
        """
    )


# ============================================================
# OFFLINE SCREEN
# ============================================================

def render_offline():

    render_html(
        """
        <div class="panel"
             style="
             text-align:center;
             padding:55px 25px;
             ">

            <div style="
                font-size:43px;
            ">
                ⚡
            </div>

            <div style="
                font-size:21px;
                font-weight:850;
                margin-top:12px;
            ">
                SkyGuard Core Offline
            </div>

            <div style="
                color:#68758a;
                font-size:11px;
                margin-top:7px;
            ">
                FastAPI is not currently reachable.
            </div>

            <div style="
                margin-top:15px;
                padding:10px 14px;
                display:inline-block;
                border-radius:9px;
                background:rgba(239,68,68,0.08);
                border:1px solid rgba(239,68,68,0.15);
                color:#fca5a5;
                font-size:10px;
                font-weight:750;
            ">
                http://127.0.0.1:8000
            </div>

            <div style="
                color:#59657a;
                font-size:10px;
                margin-top:15px;
            ">
                Start FastAPI and refresh the dashboard.
            </div>

        </div>
        """
    )


# ============================================================
# ROUTER
# ============================================================

page = st.session_state.page


if page == "Overview":

    render_overview()


elif page == "Live Monitor":

    render_live_monitor()


elif page == "Anomaly Explorer":

    render_anomaly_explorer()


elif page == "Sensor Health":

    render_sensor_health()


elif page == "Station Network":

    render_station_network()


elif page == "Data Explorer":

    render_data_explorer()


elif page == "How SkyGuard Works":

    render_how_atmos_works()


elif page == "System Status":

    render_system_status()


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <div class="atmos-footer">

        SkyGuard AI • Intelligent Anomaly Detection
        for Automatic Weather Stations

        &nbsp;•&nbsp;

        Real-Time Replay Monitoring Prototype

    </div>
    """
)


# ============================================================
# AUTO REFRESH
# ============================================================

if st.session_state.auto_refresh:

    time.sleep(
        refresh_rate
    )

    st.rerun()