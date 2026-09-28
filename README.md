# 🌦️ SkyGuard AI

### Intelligent Real-Time Anomaly Detection and Sensor Health Monitoring for Automatic Weather Stations

SkyGuard AI is an AI/ML-based intelligent monitoring system designed for **Automatic Weather Stations (AWS)**. It continuously analyses **temperature, atmospheric pressure, and relative humidity** observations to identify abnormal sensor behaviour, classify possible faults, assess sensor health, and generate explainable alerts.

The project is designed to improve the reliability of weather observations used in forecasting, disaster management, agriculture, aviation, climate monitoring, and other weather-dependent applications.

---

For the current prototype, historical AWS observations are **replayed as a real-time stream**, allowing the complete monitoring and inference pipeline to be demonstrated without requiring a direct connection to physical AWS hardware.

---

##  Key Features

###  Intelligent Anomaly Detection
Detects suspicious behaviour in:

- Temperature — `T2M`
- Relative Humidity — `RH2M`
- Atmospheric Pressure — `PS`

The system combines statistical evidence with machine-learning-based detection instead of relying only on fixed thresholds.

### Sensor Fault Classification
Sensor-level classifiers estimate the likely fault category and provide a confidence score for the prediction.

### Explainable Detection
For each sensor, the dashboard can display:

- Fault classification
- Model confidence
- Local Z-score
- Anomaly status
- Severity
- Sensor health

This gives operators evidence behind an alert instead of only returning a binary anomaly flag.

### Sensor Health Monitoring
Each sensor is assigned an operational health score that helps identify degradation and potentially unreliable sensors.

### Intelligent Alert Engine
Detected anomalies are converted into operational alerts containing:

- Affected sensor
- Severity
- Confidence
- Statistical evidence
- Sensor-health impact

###  Multi-Station Monitoring
The current prototype demonstrates monitoring across **41 Automatic Weather Stations**, with three sensor streams per station.

###  Real-Time Replay Engine
Historical observations are replayed as a live stream to demonstrate real-time monitoring and inference.

### Interactive Command Center
A Streamlit dashboard provides:

- Network Overview
- Live Monitor
- Anomaly Explorer
- Sensor Health
- Station Network
- Data Explorer
- Detection Pipeline
- System Status

---


##  Detection Pipeline

SkyGuard AI follows a seven-stage monitoring pipeline:

### 1. Data Ingestion
Automatic Weather Station observations enter the system.

### 2. Data Validation
Incoming observations are checked for data-quality problems such as missing information, timestamps, duplicates, and inconsistent records.

### 3. Feature Engineering
Raw weather observations are transformed into useful statistical and temporal features such as:

- Differences
- Percentage changes
- Rolling statistics
- Local Z-scores
- Cross-sensor relationships

### 4. Anomaly Detection
The system evaluates whether the current sensor behaviour is statistically unusual and combines this evidence with machine-learning-based detection.

### 5. Fault Classification
Sensor-specific ML classifiers estimate the likely fault type for temperature, humidity, and pressure sensors.

### 6. Sensor Health
Detection evidence is converted into an operational sensor-health score.

### 7. Explainable Alerts
The alert engine combines anomaly evidence, confidence, severity, and health information into an interpretable alert.

---

##  Machine Learning Components

The project contains multiple ML components rather than relying on a single model.

### Baseline Detection
Statistical and machine-learning baselines are used to identify unusual sensor behaviour.

### Isolation Forest
An Isolation Forest model provides an additional unsupervised anomaly-detection signal.

### Fault Classification
Separate trained classifiers are maintained for:

- Temperature sensor faults
- Relative-humidity sensor faults
- Pressure sensor faults

### Autoencoder
A neural-network autoencoder provides an additional representation-based anomaly-detection mechanism.

### Anomaly Fusion
Outputs from different detection components can be combined to produce the final sensor-level anomaly decision.

---

## Dashboard

### Network Overview
Provides a quick summary of:

- Monitored weather stations
- Active sensor streams
- Active alerts
- Average sensor health
- Latest network observation
- Primary diagnosis

### Live Monitor
Displays the current observation processed by the SkyGuard AI inference pipeline.

For every sensor, the interface presents:

```text
Current Value
Fault Classification
Confidence
Local Z-Score
Anomaly Status
Severity
Sensor Health
```

### Anomaly Explorer
Provides sensor-level anomaly evidence and visualises local Z-score behaviour against detection thresholds.

### Sensor Health
Displays the current operational condition and reliability score of each sensor.

### Station Network
Provides a searchable network view of all monitored Automatic Weather Stations.

### Data Explorer
Connects the raw weather observation with the machine-learning evidence used to evaluate it.

### How SkyGuard AI Works
Explains the complete end-to-end monitoring and inference architecture.

### System Status
Displays the operational status of the FastAPI backend and the major API endpoints used by the prototype.

---

##  System Architecture

```text
                 ┌─────────────────────────┐
                 │   Streamlit Dashboard   │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │     FastAPI Backend     │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │      Replay Engine      │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │    Inference Engine     │
                 └────────────┬────────────┘
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
     Temperature Model   Humidity Model   Pressure Model
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │   Anomaly Detection     │
                 │ + Fault Classification  │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Sensor Health Analysis  │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │      Alert Engine       │
                 └─────────────────────────┘
```

---

##  Tech Stack

### Programming
- Python

### Machine Learning
- Scikit-learn
- TensorFlow / Keras
- NumPy
- Pandas
- Joblib

### ML Techniques
- Statistical anomaly detection
- Local Z-score analysis
- Isolation Forest
- Random Forest based fault classification
- Autoencoder-based anomaly detection
- Feature engineering
- Anomaly fusion
- Sensor-health scoring

### Backend
- FastAPI
- Uvicorn

### Frontend / Dashboard
- Streamlit
- Plotly

### Development
- VS Code
- Git
- GitHub

---

##  Project Structure

```text
SkyGuard-AI/
│
├── app/
│   ├── api/
│   │   └── main.py
│   │
│   ├── dashboard/
│   │   └── dashboard.py
│   │
│   └── services/
│       ├── alert_engine.py
│       ├── data_service.py
│       ├── inference_engine.py
│       └── replay_engine.py
│
├── data/
│   ├── raw/
│   │
│   └── processed/
│       ├── autoencoder/
│       ├── baseline/
│       ├── classification/
│       ├── eda/
│       ├── fusion/
│       └── health/
│
├── ml/
│   ├── baseline/
│   │   └── baseline_anomaly.py
│   │
│   ├── classification/
│   │   ├── anomaly_fusion.py
│   │   └── fault_classifier.py
│   │
│   ├── deep_learning/
│   │   └── autoencoder.py
│   │
│   ├── features/
│   │   └── feature_engine.py
│   │
│   ├── health/
│   │   └── sensor_health.py
│   │
│   └── preprocessing/
│       ├── data_engine.py
│       ├── eda_engine.py
│       └── leakage_audit.py
│
├── models/
│   ├── atmos_autoencoder.keras
│   ├── autoencoder_scaler.joblib
│   ├── autoencoder_threshold.json
│   ├── baseline_statistics.joblib
│   ├── isolation_forest_baseline.joblib
│   ├── fault_classifier_features.json
│   ├── t2m_fault_classifier.joblib
│   ├── rh2m_fault_classifier.joblib
│   └── ps_fault_classifier.joblib
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

##  Running SkyGuard AI Locally

### 1. Clone the Repository

```bash
git clone https://github.com/SanchiSharma-ss/SkyGuard-AI.git
cd SkyGuard-AI
```

---

### 2. Create a Virtual Environment

The project has been tested with **Python 3.12**.

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

##  Start the Prototype

The prototype uses a FastAPI backend and a Streamlit frontend.

### Terminal 1 — Start FastAPI

```bash
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

The backend will be available at:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

---

### Terminal 2 — Start the Replay

On Windows PowerShell:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/replay/start" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"interval_seconds":0.5}'
```

The replay engine simulates a live AWS feed using historical observations.

---

### Terminal 3 — Start the Dashboard

```bash
python -m streamlit run app/dashboard/dashboard.py
```

Open:

```text
http://localhost:8501
```

---

## 🔌 Important API Endpoints

| Endpoint | Purpose |
|---|---|
| `/` | Backend information |
| `/health` | Service health check |
| `/statistics` | Network statistics |
| `/stations` | Available AWS stations |
| `/replay/status` | Replay-engine status |
| `/replay/current` | Current replay observation |
| `/replay/inference` | Current inference information |
| `/replay/live-result` | Latest complete SkyGuard inference result |

---

## 📊 Prototype Dataset

The current prototype operates on historical Automatic Weather Station observations containing:

- Temperature
- Relative Humidity
- Atmospheric Pressure
- Station information
- Timestamps
- Anomaly / fault information used during development and evaluation

The data is processed through separate preprocessing, feature-engineering, evaluation, and ML stages before being consumed by the real-time demonstration pipeline.

---

## Prototype Replay Mode

The current demonstration does **not claim a direct live connection to physical AWS hardware**.

Instead:

> Historical AWS observations are replayed sequentially as a real-time stream.

This makes it possible to demonstrate the complete:

```text
Observation
→ Feature Engineering
→ ML Inference
→ Fault Classification
→ Health Monitoring
→ Alert Generation
→ Dashboard
```

pipeline in a reproducible environment.

The architecture can later be connected to a real AWS data stream by replacing the replay input layer.

---

##  Scalability

SkyGuard AI is designed so that the monitoring logic is separated from the data-ingestion layer.

This allows the architecture to be extended from the current prototype toward:

- Larger AWS networks
- Continuous streaming data
- Cloud-hosted inference
- Centralised monitoring
- Multiple geographic regions
- Additional sensor-health analytics
- API integration with meteorological systems

---

## Potential Impact

Reliable weather observations are essential for:

- Weather forecasting
- Disaster preparedness
- Agriculture
- Aviation
- Climate monitoring
- Flood and storm monitoring
- Scientific research

By detecting unreliable observations before they propagate into downstream systems, SkyGuard AI can support better quality control and faster identification of malfunctioning sensors.

---

## 🔮 Future Scope

Potential extensions include:

- Integration with live AWS data feeds
- Cloud deployment
- Automated maintenance recommendations
- Station-to-station spatial consistency checks
- Weather-event-aware anomaly verification
- Advanced temporal models
- Model drift monitoring
- Automatic sensor recalibration recommendations
- Notification and escalation systems
- Geographic visualisation of network-wide anomalies

---

##  Prototype Status

SkyGuard AI is currently a functional prototype developed for demonstrating an intelligent AWS anomaly-detection and monitoring architecture.

Historical weather observations are used in replay mode to simulate real-time operation.

The system is intended as a decision-support and monitoring layer; detected anomalies should be interpreted together with meteorological and operational context.

---

## Repository

**GitHub:**  
[github.com/SanchiSharma-ss/SkyGuard-AI](https://github.com/SanchiSharma-ss/SkyGuard-AI)

---

##  License

This repository has been developed as an academic and hackathon prototype.

Please contact the project contributors before reusing the complete project or trained models for external applications.

---

<p align="center">
  <b>SkyGuard AI</b><br>
  Intelligent Weather Station Monitoring
</p>
