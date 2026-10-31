"""Unit tests for src/features.py (vibration feature extraction)."""

import numpy as np
import pytest

from src.features import (DEFECT_ORDERS, EXCLUDE_COLUMNS, FEATURE_NAMES, TARGET_FS, extract_features,
                          features_to_frame)

RNG = np.random.default_rng(0)


def test_feature_names_unique_and_leak_free():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 18
    for col in EXCLUDE_COLUMNS:
        assert col not in FEATURE_NAMES


def test_gaussian_noise_statistics():
    x = RNG.normal(0, 0.5, 2048)
    f = extract_features(x)
    assert list(f) == FEATURE_NAMES
    assert abs(f["rms"] - 0.5) < 0.03
    assert abs(f["kurtosis"] - 3.0) < 0.5  # Gaussian kurtosis ~ 3
    assert abs(f["skewness"]) < 0.3
    assert abs(sum(f[k] for k in FEATURE_NAMES if k.startswith("band_")) - 1.0) < 1e-6


def test_impulses_raise_kurtosis_and_crest_factor():
    x = RNG.normal(0, 0.1, 2048)
    y = x.copy()
    y[::180] += 3.0
    fx, fy = extract_features(x), extract_features(y)
    assert fy["kurtosis"] > fx["kurtosis"] + 3
    assert fy["crest_factor"] > fx["crest_factor"]


def test_envelope_feature_detects_periodic_impacts_at_bpfo():
    rpm = 1797.0
    bpfo = DEFECT_ORDERS["env_bpfo"] * rpm / 60
    t = np.arange(2048) / TARGET_FS
    impacts = (np.sin(2 * np.pi * bpfo * t) > 0.98).astype(float)  # impact train at BPFO
    ring = np.sin(2 * np.pi * 3500 * t)  # structural resonance inside the 2-5.5 kHz band
    x = 0.05 * RNG.normal(size=2048) + 2.0 * impacts * ring
    healthy = 0.05 * RNG.normal(size=2048)
    assert extract_features(x, TARGET_FS, rpm)["env_bpfo"] > 3 * extract_features(healthy, TARGET_FS, rpm)["env_bpfo"]


def test_dc_offset_is_ignored():
    x = RNG.normal(0, 0.3, 2048)
    a, b = extract_features(x), extract_features(x + 5.0)
    for k in FEATURE_NAMES:
        assert a[k] == pytest.approx(b[k], rel=1e-6, abs=1e-9)


def test_rejects_bad_windows():
    with pytest.raises(ValueError):
        extract_features(np.zeros(10))
    with pytest.raises(ValueError):
        extract_features(np.array([np.nan] * 512))


def test_features_to_frame():
    f = extract_features(RNG.normal(size=2048))
    df = features_to_frame(f)
    assert df.shape == (1, len(FEATURE_NAMES)) and list(df.columns) == FEATURE_NAMES
    bad = dict(f)
    bad.pop("rms")
    with pytest.raises(ValueError):
        features_to_frame(bad)
    bad = dict(f, kurtosis=float("nan"))
    with pytest.raises(ValueError):
        features_to_frame(bad)
