"""
Unit tests for model loading and prediction pipeline.
"""

import os
import sys
import joblib
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import create_features_single
from src.inference_service import InferenceService

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best_model.joblib"))


def test_model_artifact_exists():
    assert os.path.exists(MODEL_PATH), "best_model.joblib must exist."


def test_model_prediction():
    model = joblib.load(MODEL_PATH)
    sample_normal = {
        "type": "M",
        "air_temp": 298.1,
        "process_temp": 308.6,
        "rpm": 1500,
        "torque": 40.0,
        "tool_wear": 10,
    }
    feats = create_features_single(sample_normal)
    proba = model.predict_proba(feats)[0, 1]

    assert 0.0 <= proba <= 1.0


def test_inference_service_predict():
    service = InferenceService()
    sample_faulty = {
        "type": "L",
        "air_temp": 298.1,
        "process_temp": 308.6,
        "rpm": 1300,
        "torque": 65.0,
        "tool_wear": 220,
    }
    res = service.predict(sample_faulty)

    assert "failure_prob" in res
    assert "status" in res
    assert res["status"] in ["NORMAL", "WARNING", "CRITICAL"]
