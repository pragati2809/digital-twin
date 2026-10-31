# DEMO & VIVA GUIDE

**Project:** Digital Twin for Bearing Health Monitoring (HCL P_222) · **Data:** CWRU Bearing Data Center (real vibration)

## 3-minute demo
**Minute 1 - problem and architecture.** Unplanned bearing failures stop machines. The twin receives vibration over MQTT,
stores it in InfluxDB, diagnoses Normal / Inner-race / Ball / Outer-race with an ML model, and shows it live.
Pipeline: Replay publisher -> Mosquitto -> Ingest + Inference services -> InfluxDB -> Grafana + Streamlit (all in Docker).

**Minute 2 - data and ML.**
- Real accelerometer data from CWRU (drive end, 12 kHz). Normal files were recorded at 48 kHz and are resampled to 12 kHz.
- 2048-sample windows -> 18 features: RMS, kurtosis, crest factor, band energies and *envelope-spectrum peaks at the
  bearing defect frequencies* (BPFO/BPFI/ball). Labels, file IDs and shaft speed are not model inputs.
- LightGBM chosen on the **validation** set; reported once on the **test** set: fault detection F1 = 1.00, 4-class
  accuracy 100 %. Say immediately: *"this is with one recording per fault class, so fault windows come from different time
  blocks of the same recording; normal is tested on a completely unseen recording (100.mat, 3 HP)."*

**Minute 3 - live demo.**
1. Streamlit **Live Monitor** (`:8501`): the status badge and diagnosis change as the scenario plays
   (normal -> inner race -> ball -> outer race).
2. **Fault Simulator**: choose `105.mat` (inner race), show waveform, spectrum and the envelope spectrum with the BPFI peak;
   then add noise (SNR 10 dB) and show the model on `118.mat` (ball) becoming unreliable - an honest limitation.
3. **Grafana** (`:3000`): failure risk vs thresholds, class probabilities, kurtosis, defect-frequency features.

## Anticipated viva questions
**Why does a 100 % score not prove the model works in the field?** Only one recording per fault class exists here, so
fault train/test windows share the same recording (separated by time with a purge gap, no shared samples). A real test needs
other loads, fault sizes or machines. Adding the CWRU files for loads 1-3 and 14-mil faults switches the split to whole-recording hold-out.

**How did you prevent leakage?** Windows overlap 75 %, so random splitting would put near-duplicates in train and test. I split by
time block with a gap larger than a window (faults) or by whole recording (normal). Model inputs exclude file ID, label, load and RPM.
Model selection used validation data only.

**Why kurtosis and envelope analysis?** A defect produces periodic impacts that excite a structural resonance. Impacts make the
signal impulsive (kurtosis rises above 3) and the *envelope* of the high-frequency band repeats at BPFO/BPFI/ball-defect frequency.

**Why 1 - P(normal) as failure probability?** The model is 4-class; the probability of "not normal" is the natural fault score and is
thresholded on validation data. Status: NORMAL < 0.30, WARNING 0.30-0.50, CRITICAL >= 0.50.

**What do MQTT and InfluxDB do?** MQTT is a lightweight publish/subscribe transport for sensor streams. InfluxDB stores timestamped
telemetry and predictions and serves Flux window aggregations to Grafana.

**What is weak?** Ball faults (weakest signature, 27 % accuracy at 10 dB SNR in my stress test), single bearing type, lab data, and
an accelerated replay rather than a physical sensor.
