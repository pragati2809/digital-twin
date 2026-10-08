# Digital Twin for Industrial Equipment Health Monitoring

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Docker Compose](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

An end-to-end, deployable software **Digital Twin** system for industrial machine predictive maintenance and real-time health monitoring. Built for CSE (AI & ML) final-year project (HCL Project List ID **P_222**).

---

## 🎯 Architecture Diagram

```
+------------------------+      MQTT (1883)       +------------------------+
|  ai4i2020.csv Dataset  | ---------------------> |   Mosquitto Broker     |
|   (Replay Publisher)   | topic: machine/telemetry| (strange_varahamihira) |
+------------------------+                        +------------------------+
                                                    /                    \
                                                   /                      \
                                                  v                        v
                           +---------------------------+       +---------------------------+
                           |      Ingest Service       |       |     Inference Service     |
                           |   (sensor field storage)  |       |   (best_model.joblib RF)  |
                           +---------------------------+       +---------------------------+
                                         |                                  |
                                         v                                  v
                               +--------------------------------------------------+
                               |     InfluxDB 2.7 Time-Series DB (port 8086)     |
                               |      measurement: telemetry | health            |
                               +--------------------------------------------------+
                                         /                                  \
                                        /                                    \
                                       v                                      v
            +------------------------------------+          +------------------------------------+
            | Grafana Live Ops (port 3000)       |          | Streamlit Dashboard (port 8501)    |
            | Provisioned Flux Dashboards        |          | Live Monitor, What-If, Explorer    |
            +------------------------------------+          +------------------------------------+
```

---

## 📊 Dataset & Honesty Note

This project is trained and evaluated on the **AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning Repository, Dataset ID 601, CC BY 4.0).

> [!NOTE]
> **Dataset Classification**: UCI documents this dataset as a public synthetic benchmark dataset modeled on real industrial predictive maintenance behavior. It contains 10,000 tool wear records across 14 columns, with a heavily imbalanced target class (3.39% machine failure rate). In all project reports and viva presentations, it is accurately described as "a public benchmark dataset modeled on real industrial predictive-maintenance data".

---

## 🏆 Model Training & Test Set Evaluation Results

Models were trained on 70% split (7,000 samples), threshold-tuned on 15% Validation split (1,500 samples), and evaluated on the untouched **15% Test set** (1,500 samples, 51 failure cases):

| Model Candidate | Optimal Threshold | Test Precision | Test Recall | Test F1 Score | Test PR-AUC | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Balanced)** 🏆 | **0.4300** | **0.9574** | **0.8824** | **0.9184** | **0.9303** | **0.9747** |
| LightGBM (`scale_pos_weight`) | 0.5500 | 0.8519 | 0.9020 | 0.8762 | 0.9454 | **0.9909** |
| Heuristic Threshold Baseline | 0.0500 | 0.4216 | 0.8431 | 0.5621 | 0.3948 | 0.9024 |
| Isolation Forest (Unsupervised) | 0.4200 | 0.0800 | 0.7843 | 0.1452 | 0.2535 | 0.8352 |

> **Key takeaway**: Accuracy alone (99.1%) is misleading due to the 3.39% class imbalance. Our Random Forest pipeline achieves **95.74% Precision** and **88.24% Recall** (F1 = **0.9184**) on failure detection.

---

## 🚀 Quick Start & Setup Instructions (Windows 11 PowerShell)

### Prerequisites
1. Python 3.13 installed
2. Docker Desktop installed and running

### 1. Clone & Install Dependencies
```powershell
git clone https://github.com/pragati2809/digital-twin.git
cd digital-twin
pip install -r requirements.txt
```

### 2. Run Exploratory Data Analysis & Model Training
```powershell
python scripts/eda.py
python src/train.py
```
This generates `models/best_model.joblib`, `models/metadata.json`, and all plot figures under `reports/`.

### 3. Run Pytest Verification Suite
```powershell
pytest -v
```

### 4. Option A: Single-Command Docker Compose Deployment
Ensure Docker Desktop is open, then run:
```powershell
docker compose up --build -d
```

Access services:
- **Streamlit Interactive App**: [http://localhost:8501](http://localhost:8501)
- **Grafana Live Ops Dashboard**: [http://localhost:3000](http://localhost:3000) (User: `admin`, Pass: `admin`)
- **InfluxDB Admin UI**: [http://localhost:8086](http://localhost:8086) (Org: `digital-twin-org`, Bucket: `machine_data`)

### 5. Option B: Running Microservices Locally (Standalone)
If running without Docker Compose containers:
```powershell
# Terminal 1: Ingest Service
python ingest.py

# Terminal 2: Inference Service
python src/inference_service.py

# Terminal 3: Replay Publisher Stream
python src/replay_publisher.py --rate 2.0 --loop

# Terminal 4: Streamlit Dashboard
streamlit run app/streamlit_app.py
```

---

## ⚠️ Limitations & Future Work

1. **Synthetic Data Modeling**: While the AI4I 2020 dataset reflects real physical failure mechanisms (Tool Wear, Heat Dissipation, Power Overload), validation on physical measured bearing testbed data (such as the Case Western Reserve University (CWRU) Bearing Dataset) remains future work.
2. **Edge Deployment**: Compiling the Random Forest pipeline to ONNX format for microcontroller/Raspberry Pi edge inference.
