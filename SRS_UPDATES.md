# 📄 SOFTWARE REQUIREMENTS SPECIFICATION (SRS) & FINAL PROJECT REPORT UPDATES

**Project Title:** Digital Twin for Industrial Equipment Health Monitoring  
**HCL Project List ID:** P_222  
**Domain:** Industrial IoT, Machine Learning, Predictive Maintenance  
**Academic Degree:** B.Tech CSE (AI & ML)  
**Font & Style Guidelines:** Times New Roman, 12pt body text, 14pt bold section headings, 1.15 line spacing.

---

## 🏛️ REVISED SYSTEM ARCHITECTURE DIAGRAM

```
+-----------------------------------------------------------------------------------+
|                            DATA REPLAY & INGESTION LAYER                          |
|                                                                                   |
|  +------------------------+    MQTT (1883)     +-------------------------------+  |
|  |  ai4i2020.csv Dataset  | -----------------> | Mosquitto Broker (Paho v2 API)|  |
|  |   (Replay Publisher)   |  machine/telemetry | (strange_varahamihira)        |  |
|  +------------------------+                    +-------------------------------+  |
+--------------------------------------------------------|--------------------------+
                                                         |
                                                         v
+-----------------------------------------------------------------------------------+
|                        REAL-TIME ML & INFERENCE LAYER                             |
|                                                                                   |
|  +---------------------------+                 +-------------------------------+  |
|  |      Ingest Service       |                 |     Inference Service         |  |
|  | (Raw sensor field storage)|                 | (Random Forest Pipeline)      |  |
|  +---------------------------+                 +-------------------------------+  |
|                |                                               |                  |
|                v                                               v                  |
|  +-----------------------------------------------------------------------------+  |
|  |               InfluxDB 2.7 Time-Series DB (bucket: machine_data)           |  |
|  |               measurements: telemetry | health (tags: type, status)         |  |
|  +-----------------------------------------------------------------------------+  |
+--------------------------------------------------------|--------------------------+
                                                         |
                                                         v
+-----------------------------------------------------------------------------------+
|                           VISUALIZATION & ANALYTICS LAYER                         |
|                                                                                   |
|  +------------------------------------+   +------------------------------------+  |
|  | Grafana Live Ops Dashboard         |   | Streamlit Multipage Web App        |  |
|  | (Provisioned Flux panels, alerts)  |   | (Live Monitor, What-If, Explorer)  |  |
|  +------------------------------------+   +------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

---

## 📈 EXPERIMENTAL RESULTS & MODEL PERFORMANCE SECTION

### 1. Dataset Description & Preprocessing
The system was trained and evaluated on the **AI4I 2020 Predictive Maintenance Dataset** (UCI ML Repository, Dataset ID 601, CC BY 4.0). The dataset contains **10,000 tool wear records** across 14 raw variables. Machine failures account for **339 records (3.39% failure rate)**, representing a heavily imbalanced binary classification challenge.

Feature engineering produced 11 input variables:
1. `air_temp` [K]
2. `process_temp` [K]
3. `rpm` [RPM]
4. `torque` [Nm]
5. `tool_wear` [min]
6. `temp_diff = process_temp - air_temp`
7. `power_w = torque * rpm * (2 * pi / 60)`
8. `wear_torque = tool_wear * torque`
9. `type_L`, `type_M`, `type_H` (One-Hot Encoded)

**Data Leakage Prevention**: Features `UDI`, `Product ID`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`, and `Machine failure` were strictly excluded from model inputs.

### 2. Model Comparison Table (Test Set Evaluation)
Evaluated on an untouched **15% Test set** (1,500 samples, 51 true failure cases):

| Model Architecture | Optimal Threshold | Precision | Recall | F1 Score | PR-AUC | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Balanced)** 🏆 | **0.4300** | **95.74%** | **88.24%** | **0.9184** | **0.9303** | **0.9747** |
| LightGBM (`scale_pos_weight`) | 0.5500 | 85.19% | 90.20% | 0.8762 | 0.9454 | 0.9909 |
| Heuristic Threshold Baseline | 0.0500 | 42.16% | 84.31% | 0.5621 | 0.3948 | 0.9024 |
| Isolation Forest (Unsupervised) | 0.4200 | 8.00% | 78.43% | 0.1452 | 0.2535 | 0.8352 |

### 3. Confusion Matrix & Feature Importance
For the winning **Random Forest** model at decision threshold $th = 0.4300$:
- **True Negatives (TN)**: 1,447
- **False Positives (FP)**: 2 (Precision = 95.74%)
- **False Negatives (FN)**: 6 (Recall = 88.24%)
- **True Positives (TP)**: 45

**Feature Importance Hierarchy**:
1. `torque` & `wear_torque` (42.3% combined importance)
2. `tool_wear` (22.1%)
3. `power_w` (14.5%)
4. `rpm` (10.8%)
5. `temp_diff` (7.2%)

---

## 📝 SECTION-BY-SECTION SRS REPLACEMENT TEXT (17 SECTIONS)

### Section 1: Abstract
> **Replacement Text**:
> This document details the software requirements and design specifications for an industry-style Digital Twin for Industrial Equipment Health Monitoring (HCL Project List ID P_222). Designed for real-time fault detection and predictive maintenance, the software digital twin replaces simulated heuristic models with machine learning pipelines trained on the public AI4I 2020 Predictive Maintenance benchmark dataset. Sensor telemetry streams over MQTT into an InfluxDB time-series database where a Random Forest classification pipeline scores machine health in real time. Operational status is rendered via Grafana live dashboards and an interactive Streamlit web application.

### Section 2: Problem Statement
> **Replacement Text**:
> Industrial manufacturing facilities suffer severe financial losses due to unplanned equipment downtime and reactive repair strategies. Traditional monitoring systems rely on static threshold alerts that fail to capture non-linear degradation and complex interactions between temperature, rotational speed, torque, and cumulative tool wear. This project addresses the challenge by developing an automated software Digital Twin that continuously calculates failure probability, detects anomalous health degradation prior to catastrophic breakdown, and provides actionable maintenance insights.

### Section 3: Proposed Solution
> **Replacement Text**:
> The proposed solution is a microservice-based Digital Twin architecture that ingests 1.0 Hz industrial machine telemetry over MQTT using the Paho v2 API. An Ingestion Service writes sensor readings and engineered physics features (Temperature Difference, Mechanical Power, Wear-Torque Load) to InfluxDB. Concurrently, an Inference Service applies a pre-trained Random Forest model to compute failure risk probabilities and classify machine operational state into NORMAL, WARNING, or CRITICAL health tiers. Operational telemetry and model inferences are visualized through Grafana live operational dashboards and a multipage Streamlit interactive control center.

### Section 4: System Objectives
> **Replacement Text**:
> 1. To engineer a reproducible telemetry streaming pipeline capable of replaying 10,000 industrial sensor records over MQTT.
> 2. To develop domain feature transformations (mechanical power, thermal differential, wear-torque load) without target data leakage.
> 3. To train and validate supervised machine learning models achieving an F1 score $\ge 0.90$ on imbalanced industrial failure data.
> 4. To deploy automated time-series database ingestion and Grafana dashboard provisioning.
> 5. To provide a What-If simulation interface allowing plant engineers to test physical parameter boundaries.

### Section 5: Target Audience & Scope
> **Replacement Text**:
> The target audience includes industrial plant operations managers, reliability engineers, maintenance technicians, and CSE (AI & ML) project evaluators. The system scope encompasses real-time MQTT streaming, time-series storage, machine learning inference, automated alerting, and multi-tier web dashboard visualization.

### Section 6: Technology Stack
> **Replacement Text**:
> - **Programming Language**: Python 3.13.7
> - **Machine Learning Libraries**: Scikit-Learn 1.6.1, LightGBM 4.0.0, Joblib 1.3.0, Pandas 2.2.3, NumPy 2.2.5
> - **Visualization**: Streamlit 1.30.0, Matplotlib 3.11.2, Seaborn 0.13.2, Grafana (Latest Docker)
> - **Data Ingestion & Protocols**: Eclipse Mosquitto MQTT 2.0 (Port 1883), Paho MQTT v2 API
> - **Time-Series Database**: InfluxDB 2.7 (Port 8086, Flux Query Engine)
> - **Containerization**: Docker Desktop & Docker Compose 3.8
> - **Testing Framework**: Pytest 9.1.1

### Section 7: Software Requirements
> **Replacement Text**:
> - **SR-1**: The system shall subscribe to MQTT topic `machine/telemetry` and process incoming JSON payloads within 50 milliseconds.
> - **SR-2**: The feature engineering module shall compute `temp_diff`, `power_w`, `wear_torque`, and one-hot encoded `type` features without importing target leakage fields.
> - **SR-3**: The ML inference pipeline shall output a failure probability $p \in [0, 1]$, predicted binary failure class, and status tier (`NORMAL`, `WARNING`, `CRITICAL`).
> - **SR-4**: InfluxDB measurement `telemetry` shall store raw sensor fields and derived features; measurement `health` shall store inference scores and status codes.
> - **SR-5**: Grafana and Streamlit dashboards shall auto-refresh telemetry views every 5 seconds.

### Section 8: Hardware & Infrastructure Requirements
> **Replacement Text**:
> - **Processor**: Quad-core x86_64 CPU (2.0 GHz or higher)
> - **RAM**: 8 GB RAM minimum (16 GB recommended)
> - **Storage**: 10 GB free disk space (Docker image storage allocated to secondary volume if needed)
> - **Network**: Local host loopback interfaces on ports 1883, 8086, 3000, 8501

### Section 9: System Architecture & Dataflow
> **Replacement Text**:
> The system follows a decoupled event-driven microservices architecture:
> `CSV Replay -> MQTT Broker -> Ingest Service -> InfluxDB -> Grafana`
> `                          \-> Inference Service -> InfluxDB + MQTT Health Alerts -> Streamlit`
> All services run in isolated Docker containers communicating across a shared bridge network.

### Section 10: Database Schema & Design
> **Replacement Text**:
> - **Bucket**: `machine_data` (Org: `digital-twin-org`)
> - **Measurement `telemetry`**:
>   - *Tags*: `type` (`L`, `M`, `H`)
>   - *Fields*: `air_temp`, `process_temp`, `rpm`, `torque`, `tool_wear`, `temp_diff`, `power_w`, `wear_torque`, `label`, `vibration`, `temperature`
> - **Measurement `health`**:
>   - *Tags*: `type`, `status` (`NORMAL`, `WARNING`, `CRITICAL`)
>   - *Fields*: `failure_prob`, `anomaly_score`, `predicted_failure`, `status_code` (0, 1, 2), `label`

### Section 11: Machine Learning Methodology
> **Replacement Text**:
> Machine learning model selection employed a stratified 70/15/15 train/validation/test split with seed 42. Four candidate architectures were evaluated: a domain Heuristic Baseline, an unsupervised Isolation Forest, a balanced Random Forest, and a weighted LightGBM classifier. Decision thresholds were optimized on the validation split to maximize F1 score while maintaining a target recall $\ge 85\%$ on the imbalanced 3.39% failure class. Random Forest was selected as the best pipeline, yielding 95.74% precision, 88.24% recall, and 0.9184 F1 score on the test set.

### Section 12: User Interface Design
> **Replacement Text**:
> The user interface comprises two complementary platforms:
> 1. **Grafana Operational Dashboard**: Dark-mode operational monitor displaying gauge indicators, live sensor time series, failure probability graphs, and actual vs predicted comparison tables.
> 2. **Streamlit Interactive Application**: Multipage control panel featuring a Live Monitor tab, Model Performance tab (PR/ROC curves, confusion matrix, comparison table), What-If Predictor tab (interactive sensor sliders for real-time inference), and Dataset Explorer tab.

### Section 13: System Testing & Verification
> **Replacement Text**:
> Verification was performed using a Pytest automated test suite:
> - `test_features.py`: Validates mathematical correctness of engineered features (`temp_diff`, `power_w`, `wear_torque`) and verifies strict exclusion of leakage columns.
> - `test_model.py`: Verifies existence of `best_model.joblib` artifact and tests end-to-end prediction execution.
> - `test_ingest.py`: Tests MQTT JSON payload deserialization and InfluxDB Point generation.
> All 6 unit tests passed cleanly with 0 errors.

### Section 14: Deployment & DevOps Guidelines
> **Replacement Text**:
> System deployment is containerized using Docker Compose (`docker-compose.yml`). The stack defines services for Mosquitto MQTT, InfluxDB 2.7, Grafana, Replay Publisher, Ingestion Service, Inference Service, and Streamlit. Single-command execution `docker compose up --build -d` provisions all environment variables, mounted volumes, and datasource configurations without manual user intervention.

### Section 15: Limitations & Honest Disclosure
> **Replacement Text**:
> **Data Origin Disclosure**: As documented by the UCI Machine Learning Repository, the AI4I 2020 Predictive Maintenance Dataset is a synthetic benchmark dataset modeled on real industrial machine behavior. While it accurately reflects physical failure modes (Tool Wear, Heat Dissipation, Power Overload), validation on physical measured bearing testbed data (such as the CWRU bearing dataset) remains future work.

### Section 16: Future Scope
> **Replacement Text**:
> 1. Integration with physical accelerometer and thermocouple IoT hardware via ESP32/Raspberry Pi microcontrollers.
> 2. Model quantization and ONNX conversion for edge deployment on low-power industrial gateway devices.
> 3. Implementation of Remaining Useful Life (RUL) continuous regression modeling using Recurrent Neural Networks (LSTM/GRU).

### Section 17: References
> **Replacement Text**:
> 1. Matzka, S. (2020). Explainable Artificial Intelligence for Predictive Maintenance Applications. *Third International Conference on Artificial Intelligence for Industries (AI4I)*, pp. 69-74. DOI: 10.1109/AI4I49448.2020.00023.
> 2. UCI Machine Learning Repository (2020). *AI4I 2020 Predictive Maintenance Dataset* (ID 601). Available at: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset.
> 3. InfluxData (2024). *InfluxDB 2.7 Flux Language Reference & Time-Series Best Practices*.
> 4. Eclipse Foundation (2024). *Mosquitto MQTT Broker v2.0 Documentation*.
> 5. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, pp. 2825-2830.
