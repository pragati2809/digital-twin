"""Tests for the trained model artifact and the inference service."""

import json
import os

import joblib
import numpy as np
import pytest

from src.data import CLASS_NAMES
from src.features import FEATURE_NAMES
from src.inference_service import InferenceService
from src.replay_publisher import build_payload, load_window_table

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT, "models", "best_model.joblib")


@pytest.fixture(scope="module")
def test_windows():
    df = load_window_table()
    return df[df["split"] == "test"].reset_index(drop=True)


def test_model_artifact_and_metadata():
    assert os.path.exists(MODEL_PATH)
    bundle = joblib.load(MODEL_PATH)
    assert bundle["feature_names"] == FEATURE_NAMES and bundle["class_names"] == CLASS_NAMES
    meta = json.load(open(os.path.join(ROOT, "models", "metadata.json")))
    assert 0.0 < meta["optimal_threshold"] < 1.0
    assert meta["status_thresholds"]["normal_max"] <= meta["status_thresholds"]["warning_max"]


def test_probabilities_are_valid(test_windows):
    bundle = joblib.load(MODEL_PATH)
    proba = bundle["pipeline"].predict_proba(test_windows[FEATURE_NAMES])
    assert proba.shape == (len(test_windows), 4)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_held_out_accuracy(test_windows):
    bundle = joblib.load(MODEL_PATH)
    pred = bundle["pipeline"].predict(test_windows[FEATURE_NAMES])
    assert (pred == test_windows["class_id"].to_numpy()).mean() > 0.95


def test_inference_service_each_class(test_windows):
    svc = InferenceService(connect_influx=False)
    for cid, name in enumerate(CLASS_NAMES):
        row = test_windows[test_windows["class_id"] == cid].iloc[len(test_windows[test_windows["class_id"] == cid]) // 2]
        res = svc.predict(build_payload(row, 1))
        assert res["fault_type"] == name
        assert res["status"] in ("NORMAL", "WARNING", "CRITICAL")
        assert (res["status"] == "NORMAL") == (cid == 0)
        assert 0.0 <= res["failure_prob"] <= 1.0


def test_process_telemetry_publishes_and_tracks_state(test_windows):
    class FakeMqtt:
        def __init__(self):
            self.sent = []

        def publish(self, topic, payload, qos=0):
            self.sent.append((topic, json.loads(payload)))

    svc = InferenceService(connect_influx=False)
    fake = FakeMqtt()
    fault_row = test_windows[test_windows["class_id"] == 1].iloc[0]
    svc.process_telemetry(fake, json.dumps(build_payload(fault_row, 7)).encode())
    topic, msg = fake.sent[0]
    assert topic == "machine/health" and msg["status"] == "CRITICAL" and msg["fault_type"] == "inner_race"
    assert svc.twin_state.total_alerts == 1 and svc.twin_state.tick_count == 7


def test_bad_payload_does_not_crash_service():
    svc = InferenceService(connect_influx=False)

    class FakeMqtt:
        def publish(self, *a, **k):
            raise AssertionError("must not publish for invalid telemetry")

    svc.process_telemetry(FakeMqtt(), b'{"t": 1, "rms": 0.1}')
    svc.process_telemetry(FakeMqtt(), b"not json")
