# 🎬 DEMO & VIVA PRESENTATION GUIDE

**Project Title:** Digital Twin for Industrial Equipment Health Monitoring  
**Student:** CSE (AI & ML) Final Year  
**HCL Project List ID:** P_222  

---

## ⏱️ 3-Minute Viva Demonstration Script

### Minute 1: Problem & System Architecture (0:00 - 1:00)
- **Opening**: *"Good morning examiners. My project is a deployable Digital Twin software system for real-time industrial equipment health monitoring."*
- **Problem**: *"Unplanned machine downtime causes massive industrial losses. Traditional threshold alerts trigger too late. Our Digital Twin streams real-time sensor telemetry over MQTT, ingests it into InfluxDB, and scores failure probability live using a trained Random Forest classifier."*
- **Architecture**: *"The pipeline consists of 5 containerized services: Replay Publisher -> Mosquitto MQTT Broker -> Ingest & ML Inference Service -> InfluxDB -> Grafana Live Ops & Streamlit Web Application."*

### Minute 2: Real-Data ML Performance & Dataset (1:00 - 2:00)
- **Dataset**: *"We trained on the UCI AI4I 2020 Predictive Maintenance Dataset (10,000 records). It has a 3.39% failure rate."*
- **Feature Engineering**: *"We engineered domain features including Temperature Difference, Mechanical Power ($P = \tau \cdot \omega$), and Wear-Torque Load ($ToolWear \cdot Torque$). Crucially, we excluded label-leaking columns like TWF, HDF, PWF to prevent data leakage."*
- **ML Quality**: *"Due to extreme class imbalance, accuracy is misleading. Our Random Forest model achieved **95.74% Precision** and **88.24% Recall** (F1 = **0.9184**) on the untouched 15% Test set, outperforming Isolation Forest and LightGBM."*

### Minute 3: Live Demo & What-If Simulation (2:00 - 3:00)
- **Streamlit Demo**: Open `http://localhost:8501`.
  - Show **Live Monitor**: Point out status badge (`NORMAL` vs `CRITICAL`) changing in real time as dataset rows replay.
  - Show **What-If Predictor**: Slide `Tool Wear` to 220 min and `Torque` to 65 Nm; demonstrate failure risk spiking to >90% and triggering a `CRITICAL` alert status.
  - Show **Grafana Dashboard**: Open `http://localhost:3000` to show provisioned Flux time-series graphs.

---

## ❓ Top Anticipated Viva Questions & Answers

### Q1: Why use F1-score and PR-AUC instead of Accuracy?
> **Answer**: In our dataset of 10,000 records, only 339 (3.39%) are failure cases. A naive model that predicts "Normal" 100% of the time would get 96.61% accuracy, but would catch 0 failures. Precision, Recall, and F1 score specifically measure performance on the rare failure class.

### Q2: How did you handle data leakage in feature engineering?
> **Answer**: The dataset includes columns `TWF`, `HDF`, `PWF`, `OSF`, `RNF` which indicate the specific mode of failure *after* failure occurs. If included in ML inputs, the model would achieve artificial 100% accuracy. We strictly dropped `UDI`, `Product ID`, and all 5 failure mode flags from model training.

### Q3: What is the role of MQTT and InfluxDB in a Digital Twin?
> **Answer**: MQTT is a lightweight publish-subscribe protocol ideal for high-frequency IoT sensor streaming. InfluxDB is a time-series database optimized for storing append-only sensor timestamps and executing fast Flux window aggregations.

### Q4: How does the Digital Twin classify machine status (NORMAL / WARNING / CRITICAL)?
> **Answer**: The ML model computes a continuous failure probability $p \in [0, 1]$. Using validation threshold tuning, $p < 0.25$ is categorized as `NORMAL`, $0.25 \le p < 0.43$ as `WARNING`, and $p \ge 0.43$ as `CRITICAL`.

### Q5: Is the AI4I 2020 dataset real field data?
> **Answer**: UCI documents the AI4I 2020 dataset as a synthetic benchmark dataset modeled on real industrial predictive maintenance behavior. In our project report limitations, we note that validating on physical measured testbed data (such as the CWRU bearing dataset) is future scope.
