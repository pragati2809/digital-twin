"""
Unit tests for src/features.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import create_features_df, create_features_single, ENGINEERED_FEATURE_NAMES, EXCLUDE_COLUMNS


def test_create_features_single():
    sample_raw = {
        "type": "L",
        "air_temp": 298.1,
        "process_temp": 308.6,
        "rpm": 1551,
        "torque": 42.8,
        "tool_wear": 14,
    }
    arr = create_features_single(sample_raw)
    assert isinstance(arr, pd.DataFrame)
    assert arr.shape == (1, len(ENGINEERED_FEATURE_NAMES))

    # temp_diff = 308.6 - 298.1 = 10.5
    assert abs(arr.iloc[0]["temp_diff"] - 10.5) < 1e-4

    # Type L should have type_L=1, type_M=0, type_H=0
    assert arr.iloc[0]["type_L"] == 1.0
    assert arr.iloc[0]["type_M"] == 0.0
    assert arr.iloc[0]["type_H"] == 0.0


def test_create_features_df():
    data = {
        "UDI": [1, 2],
        "Product ID": ["L47181", "M14860"],
        "Type": ["L", "M"],
        "Air temperature [K]": [298.1, 298.2],
        "Process temperature [K]": [308.6, 308.7],
        "Rotational speed [rpm]": [1551, 1408],
        "Torque [Nm]": [42.8, 46.3],
        "Tool wear [min]": [14, 0],
        "Machine failure": [0, 0],
        "TWF": [0, 0],
        "HDF": [0, 0],
        "PWF": [0, 0],
        "OSF": [0, 0],
        "RNF": [0, 0],
    }
    df = pd.DataFrame(data)
    feats_df = create_features_df(df)

    assert feats_df.shape == (2, len(ENGINEERED_FEATURE_NAMES))
    assert list(feats_df.columns) == ENGINEERED_FEATURE_NAMES

    # Ensure leakage columns are NOT present in engineered features
    for col in EXCLUDE_COLUMNS:
        assert col not in feats_df.columns
