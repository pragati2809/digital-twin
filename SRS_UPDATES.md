# SRS & FINAL REPORT UPDATES (CWRU version)

This replaces the earlier AI4I-based text. Everything below matches the code in this repository.

## Revised architecture
```
data/cwru/*.mat -> Replay Publisher --MQTT machine/telemetry--> Mosquitto
Mosquitto -> Ingest service    -> InfluxDB measurement 'telemetry' (tag: machine)
Mosquitto -> Inference service -> InfluxDB measurement 'health' (tags: machine, status, fault_type)
                               -> MQTT machine/health
InfluxDB -> Grafana (provisioned Flux dashboard) and Streamlit (live monitor, analytics, fault simulator)
```

## Experimental results
### Dataset and preprocessing
CWRU Bearing Data Center, drive-end accelerometer. 7 recordings: normal 97-100 (loads 0-3 HP, 48 kHz) and 0.007-inch
faults at 0 HP (12 kHz): inner race 105, ball 118, outer race 130. Normal files are anti-alias resampled to 12 kHz.
The loader selects the channel by recording ID (99.mat also holds stray X098_* arrays and an `ans` variable) and falls back
to nominal speeds when RPM is absent (98.mat). 2048-sample windows, stride 512, give 1,496 labelled windows (+24 dropped in purge gaps).

### Features (18)
RMS, peak-to-peak, crest factor, kurtosis, skewness, shape factor, impulse factor, zero-crossing rate, spectral centroid,
six band-energy ratios (0-6 kHz) and Hilbert-envelope spectrum amplitudes at BPFO (3.585x), BPFI (5.415x) and ball-defect (4.714x)
shaft orders relative to the envelope noise floor.

### Split protocol
Normal: recording level (97, 98 train; 99 validation; 100 test). Faults (one recording each): time blocks 60/20/20 with a purge
gap of 4 windows. Selection and threshold tuning on validation; test used once.

### Test results (353 windows: 234 normal, 40 inner, 40 ball, 39 outer)
| Model | Precision | Recall | F1 | ROC-AUC | 4-class acc. |
| :-- | :-: | :-: | :-: | :-: | :-: |
| LightGBM (deployed) | 1.000 | 1.000 | 1.000 | 1.000 | 100 % |
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | 100 % |
| Logistic Regression | 1.000 | 1.000 | 1.000 | 1.000 | 100 % |
| Rule baseline | 0.952 | 1.000 | 0.975 | 0.996 | n/a |
| Isolation Forest (normal-only) | 0.832 | 1.000 | 0.908 | 0.966 | n/a |

Noise stress test (`reports/noise_robustness.csv`): ball-fault accuracy 97.5 % (30 dB), 77.5 % (20 dB), 27.5 % (10 dB), 5 % (0 dB);
other classes remain at 100 %.

## Section replacements
**1 Abstract.** A software digital twin streams CWRU bearing vibration over MQTT into InfluxDB, where an ML model diagnoses
Normal / Inner-race / Ball / Outer-race conditions from 18 vibration features; Grafana and Streamlit present the state.

**2 Problem.** Fixed-threshold vibration alarms react late and cannot name the failing component.

**3 Solution.** Windowed vibration feature extraction (time, spectral, envelope at defect frequencies), a 4-class model, and a
streaming MQTT/InfluxDB twin with live dashboards.

**4 Objectives.** Real-data fault diagnosis; leak-aware evaluation; real-time inference; reproducible containerised deployment.

**5 Scope.** Single drive-end sensor, one bearing type, replay of recorded data (no physical sensor).

**6 Technology.** Python 3.13, NumPy/SciPy, scikit-learn, LightGBM, Paho MQTT, Mosquitto, InfluxDB 2.7, Grafana, Streamlit, Docker Compose.

**7-8 Requirements.** `requirements.txt`; Docker with Compose v2; about 1 GB RAM for the full stack.

**9 Dataflow.** See architecture; one MQTT message = one 2048-sample window summarised as 18 features plus metadata.

**10 Schema.** `telemetry`: tag `machine`; fields 18 features, `rpm`, `label`, `true_class`, `source_file`.
`health`: tags `machine`, `status`, `fault_type`; fields `failure_prob`, `anomaly_score`, `predicted_failure`, `status_code`,
`pred_class`, `p_normal`, `p_inner_race`, `p_ball`, `p_outer_race`, `label`, `true_class`, `rpm`, and echoed `rms`, `kurtosis`,
`crest_factor`, `peak_to_peak`, `env_bpfo`, `env_bpfi`, `env_bsf`.

**11 ML method.** See features and split protocol. `failure_prob = 1 - P(normal)`. Alarm threshold = centre of the validation F1 plateau;
NORMAL < 0.6 x threshold <= WARNING < threshold <= CRITICAL.

**12 UI.** Streamlit: Live Monitor, Model Analytics, Fault Simulator, Data Explorer, System Health. Grafana: 11-panel dashboard.

**13 Testing.** 28 pytest tests: feature maths, loader key/RPM handling, split leakage, model artifact, inference service,
ingest payloads, replay scenario. Plus an end-to-end MQTT run (replay -> inference) and a noise stress test.

**14 Deployment.** `docker compose up --build -d`; model trained at image build; healthchecks gate start-up; MQTT/InfluxDB bound to localhost.

**15 Limitations (disclose).** One recording per fault class and one fault size: fault test windows come from the same recordings as
training (time-separated). Lab data, accelerated replay, ball faults weakest under noise. Flux dashboard queries and Docker build were
not executed in the development environment.

**16 Future scope.** Add CWRU loads 1-3 and 14/21-mil faults for recording-level and cross-load evaluation; domain adaptation to other
machines; remaining-useful-life estimation; edge deployment (ONNX).

**17 References.** Case Western Reserve University Bearing Data Center, https://engineering.case.edu/bearingdatacenter ;
Smith & Randall (2015), *Rolling element bearing diagnostics using the CWRU data: a benchmark study*, MSSP 64-65, 100-131.
