"""
Vibration feature extraction for bearing condition monitoring (CWRU drive-end sensor).

A *window* of raw acceleration (2048 samples @ 12 kHz) is turned into 18 features:

  time domain   rms, peak_to_peak, crest_factor, kurtosis, skewness, shape_factor,
                impulse_factor, zero_cross_rate
  frequency     spectral_centroid and the share of spectral energy in six bands
  envelope      amplitude of the Hilbert-envelope spectrum at the bearing defect
                frequencies (BPFO / BPFI / ball-defect) relative to the noise floor

Nothing here uses labels, file names or recording IDs, so there is no target leakage.
`rpm` is only used to locate the defect frequencies and is NOT a model input.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import skew as _skew

TARGET_FS = 12000

BAND_EDGES = [0, 500, 1000, 2000, 3000, 4000, 6000]
BAND_NAMES = [f"band_{lo}_{hi}" for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])]

TIME_FEATURES = [
    "rms", "peak_to_peak", "crest_factor", "kurtosis", "skewness",
    "shape_factor", "impulse_factor", "zero_cross_rate",
]
FREQ_FEATURES = ["spectral_centroid"] + BAND_NAMES
ENVELOPE_FEATURES = ["env_bpfo", "env_bpfi", "env_bsf"]

FEATURE_NAMES = TIME_FEATURES + FREQ_FEATURES + ENVELOPE_FEATURES
ENGINEERED_FEATURE_NAMES = FEATURE_NAMES  # backwards-compatible alias

# Bearing defect frequencies as multiples of shaft speed (SKF 6205-2RS drive-end bearing).
DEFECT_ORDERS = {"env_bpfo": 3.5848, "env_bpfi": 5.4152, "env_bsf": 4.7135}

# Columns that must never be fed to a model.
EXCLUDE_COLUMNS = [
    "label", "class_id", "class_name", "true_class", "source_file", "file_id",
    "win_idx", "split", "load_hp", "rpm", "t",
]

_ENV_SOS = sps.butter(4, [2000, 5500], btype="bandpass", fs=TARGET_FS, output="sos")


def _envelope_features(x: np.ndarray, fs: float, rpm: float) -> Dict[str, float]:
    env = np.abs(sps.hilbert(sps.sosfiltfilt(_ENV_SOS, x)))
    env = env - env.mean()
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env)))) / len(env)
    freqs = np.fft.rfftfreq(len(env), d=1.0 / fs)
    ref_mask = (freqs >= 10) & (freqs <= 600)
    noise = float(np.median(spec[ref_mask])) + 1e-12
    shaft_hz = rpm / 60.0
    out = {}
    for name, order in DEFECT_ORDERS.items():
        best = 0.0
        for harmonic in (1, 2):
            f0 = order * shaft_hz * harmonic
            m = (freqs >= f0 * 0.95) & (freqs <= f0 * 1.05)
            if m.any():
                best = max(best, float(spec[m].max()))
        out[name] = best / noise
    return out


def extract_features(window: np.ndarray, fs: float = TARGET_FS, rpm: float = 1797.0) -> Dict[str, float]:
    """Feature dict (keys == FEATURE_NAMES) for one window of raw acceleration."""
    x = np.asarray(window, dtype=np.float64).ravel()
    if x.size < 256 or not np.all(np.isfinite(x)):
        raise ValueError("Window must contain >= 256 finite samples")
    x = x - x.mean()

    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    mean_abs = float(np.mean(np.abs(x))) + 1e-12
    rms_safe = rms + 1e-12

    feats: Dict[str, float] = {
        "rms": rms,
        "peak_to_peak": float(x.max() - x.min()),
        "crest_factor": peak / rms_safe,
        "kurtosis": float(_kurtosis(x, fisher=False)),  # Gaussian noise ~ 3
        "skewness": float(_skew(x)),
        "shape_factor": rms / mean_abs,
        "impulse_factor": peak / mean_abs,
        "zero_cross_rate": float(np.mean(np.abs(np.diff(np.sign(x))) > 0)),
    }

    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    total = float(spec.sum()) + 1e-12
    feats["spectral_centroid"] = float((freqs * spec).sum() / total)
    for name, lo, hi in zip(BAND_NAMES, BAND_EDGES[:-1], BAND_EDGES[1:]):
        m = (freqs >= lo) & (freqs < hi) if hi < fs / 2 else (freqs >= lo)
        feats[name] = float(spec[m].sum() / total)

    feats.update(_envelope_features(x, fs, rpm))
    return {k: feats[k] for k in FEATURE_NAMES}


def features_to_frame(feats: Mapping[str, float]) -> pd.DataFrame:
    """Single feature dict (e.g. an MQTT payload) -> 1-row DataFrame in model column order."""
    missing = [k for k in FEATURE_NAMES if k not in feats]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    row = [float(feats[k]) for k in FEATURE_NAMES]
    if not np.all(np.isfinite(row)):
        raise ValueError("Non-finite feature value in payload")
    return pd.DataFrame([row], columns=FEATURE_NAMES)


def create_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """Select the model input columns from a window table, in canonical order."""
    return df[FEATURE_NAMES].astype(float)
