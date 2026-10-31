# Digital Twin for Bearing Health Monitoring (CWRU vibration data)

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Docker Compose](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com/)

A deployable software **Digital Twin** that streams bearing vibration over MQTT, stores it in InfluxDB,
diagnoses the bearing state with a trained ML model in real time, and shows it in Grafana and Streamlit.
CSE (AI & ML) final-year project, HCL Project List ID **P_222**.

The model is trained on **real measured vibration** from the Case Western Reserve University (CWRU)
Bearing Data Center: drive-end accelerometer, 12 kHz, classes **Normal / Inner-race / Ball / Outer-race**.

```
 CWRU .mat -> windows (2048 @ 12 kHz) -> 18 vibration features
                    |
        Replay Publisher --MQTT machine/telemetry--> Mosquitto
                                                  /            \
                                         Ingest service    Inference service (4-class model)
                                                  \            /      -> MQTT machine/health
                                                   InfluxDB 2.7  (telemetry | health)
                                                  /            \
                                         Grafana :3000      Streamlit :8501
```

## What the model sees

Each window of 2048 samples (0.17 s) becomes 18 features (`src/features.py`): time-domain statistics
(RMS, peak-to-peak, crest factor, **kurtosis**, skewness, shape/impulse factor, zero-crossing rate),
spectral centroid and six band-energy ratios, and **envelope-spectrum amplitudes at the bearing defect
frequencies** (BPFO, BPFI, ball defect; computed from shaft speed). Shaft speed itself and all file/label
columns are **not** model inputs, so there is no shortcut through file identity.

The deployed model outputs 4-class probabilities. `failure_prob = 1 - P(normal)`; status is
`NORMAL < 0.30 <= WARNING < 0.50 <= CRITICAL` (derived from the validation-tuned alarm threshold).

## Results (held-out test split)

| Model | Test precision | Test recall | Test F1 | ROC-AUC | 4-class accuracy |
| :-- | :-: | :-: | :-: | :-: | :-: |
| **LightGBM** (deployed, chosen on validation) | 1.000 | 1.000 | 1.000 | 1.000 | 100 % |
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | 100 % |
| Logistic Regression | 1.000 | 1.000 | 1.000 | 1.000 | 100 % |
| Rule baseline (rms/kurtosis/crest/impulse limits) | 0.952 | 1.000 | 0.975 | 0.996 | n/a |
| Isolation Forest (trained on normal only) | 0.832 | 1.000 | 0.908 | 0.966 | n/a |

Test set: 353 windows (234 normal, 40 inner-race, 40 ball, 39 outer-race). Full table: `reports/model_comparison.csv`.

### Read these numbers honestly

* **Perfect scores are expected on CWRU and do not prove field performance.** The repository has only
  **one recording per fault class** (7-mil fault, 0 HP: files 105, 118, 130). Fault train/validation/test windows
  therefore come from different *time blocks of the same recording* (75 %-overlapping windows, purged gap
  so no sample is shared). That is leak-free at the sample level but **not** a recording-level or
  cross-load test.
* The **normal** class *is* tested at recording level: 97 and 98 train, 99 validate, **100 (3 HP, a load never
  seen in training) is the test set**, and it is classified perfectly.
* **Noise stress test** (`python scripts/noise_robustness.py`): inner/outer-race and normal stay at 100 % down to
  0 dB SNR, but **ball faults degrade quickly** (97.5 % at 30 dB, 77.5 % at 20 dB, 27.5 % at 10 dB). Ball faults
  have the weakest signature and are the class to improve first.
* Windows are replayed in an accelerated demo (1 window/s represents 0.17 s of vibration).

**To get a proper recording-level evaluation**, download more files from the
[CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file) (12k drive-end:
the same faults at 1-3 HP and the 14-mil faults) into `data/cwru/` and run `python src/train.py`. IDs listed in
`src/data.py` (`RECORDING_REGISTRY`) are picked up automatically; verify them against the CWRU tables. Once a class
has >= 3 recordings the split switches to whole-recording hold-out automatically.

## Quick start

### 0. Install (Python 3.13 recommended)
```bash
pip install -r requirements.txt
```

### 1. Train, test (optional: artifacts are already included)
```bash
python scripts/eda.py        # plots + summary in reports/
python src/train.py          # models/best_model.joblib, models/metadata.json, reports/*
python -m pytest -v          # 28 tests
```

### 2. Run everything with Docker (recommended)
```bash
cp .env.example .env         # then edit the secrets (see "Deploying")
docker compose up --build -d
```
| Service | URL |
| :-- | :-- |
| Streamlit app | http://localhost:8501 |
| Grafana (admin / `GRAFANA_ADMIN_PASSWORD`, default `admin`) | http://localhost:3000 |
| InfluxDB UI (localhost only) | http://localhost:8086 |

The image trains the model during `docker build` so the artifact always matches the container's library versions.
Within ~20 s of start-up the replay publisher streams the demo scenario
(normal -> inner race -> normal -> ball -> normal -> outer race, held-out test windows only).

### 3. Or run the services locally (needs Mosquitto + InfluxDB, e.g. `docker compose up mosquitto influxdb`)
```bash
python ingest.py
python src/inference_service.py
python src/replay_publisher.py --rate 2.0
streamlit run app/streamlit_app.py
```
The Streamlit app also works with **no** infrastructure: it falls back to an offline replay of held-out CWRU
windows scored by the real model and says so in the UI.

## Streamlit pages
Live Monitor · Model Analytics · **Fault Simulator** (pick any real window, add noise, see waveform, spectrum,
envelope spectrum and the diagnosis) · Data Explorer · System Health.

## Deploying (VM / server)
1. `cp .env.example .env` and set `INFLUX_TOKEN`, `INFLUX_ADMIN_PASSWORD`, `GRAFANA_ADMIN_PASSWORD`.
   The InfluxDB volume keeps the *first* token/password; run `docker compose down -v` once if you change them later.
2. `docker compose up --build -d`, then `docker compose ps` (all `healthy`/`running`) and `docker compose logs -f inference`.
3. MQTT (1883) and InfluxDB (8086) are bound to `127.0.0.1` on purpose (the broker is anonymous). Only Grafana (3000)
   and Streamlit (8501) are published; put them behind HTTPS / a reverse proxy for public access.

## Repository layout
```
src/data.py              CWRU loader (key lookup by recording ID, 48k->12k resampling, windows, leak-aware split)
src/features.py          18 vibration features
src/train.py             model comparison, threshold tuning, artifacts, plots
src/replay_publisher.py  MQTT replay of held-out windows        ingest.py   MQTT -> InfluxDB 'telemetry'
src/inference_service.py MQTT -> model -> InfluxDB 'health' + MQTT machine/health
app/streamlit_app.py     dashboard                              grafana/    provisioned datasource + dashboard
tests/                   28 pytest tests                        scripts/    eda, noise_robustness, run_all.{ps1,sh}
data/ai4i2020.csv        legacy dataset from the earlier version (no longer used)
```

## Limitations
* One recording per fault class and a single fault size in the shipped data (see above); lab data, not field data.
* Single drive-end sensor, fixed bearing (SKF 6205); defect frequencies are specific to that bearing.
* Ball faults are the weakest class under noise.
* Not yet verified by me: the Flux queries in the Grafana dashboard against a live InfluxDB, and the Docker build
  (no Docker in the environment where this was developed). The Python MQTT pipeline was tested end to end.
