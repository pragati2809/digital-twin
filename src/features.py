"""
Feature engineering module for AI4I 2020 Predictive Maintenance Dataset.

Transforms raw sensor fields into engineered features while strictly excluding
target-leakage columns (UDI, Product ID, TWF, HDF, PWF, OSF, RNF).
"""

import math
from typing import Dict, Any, List, Union
import numpy as np
import pandas as pd

# List of input raw fields expected from telemetry stream or CSV
RAW_FEATURE_COLUMNS = [
    "type",  # 'L', 'M', or 'H'
    "air_temp",  # Kelvin
    "process_temp",  # Kelvin
    "rpm",  # Rotational speed [rpm]
    "torque",  # Torque [Nm]
    "tool_wear",  # Tool wear [min]
]

# Engineered feature names produced by the transformer
ENGINEERED_FEATURE_NAMES = [
    "air_temp",
    "process_temp",
    "rpm",
    "torque",
    "tool_wear",
    "temp_diff",
    "power_w",
    "wear_torque",
    "type_L",
    "type_M",
    "type_H",
]

# Columns that MUST be excluded from ML inputs to avoid target leakage
EXCLUDE_COLUMNS = [
    "UDI",
    "udi",
    "Product ID",
    "product_id",
    "Machine failure",
    "machine_failure",
    "label",
    "TWF",
    "twf",
    "HDF",
    "hdf",
    "PWF",
    "pwf",
    "OSF",
    "osf",
    "RNF",
    "rnf",
]


def create_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms a DataFrame with raw or standardized columns into a feature DataFrame
    suitable for model training and prediction.

    Accepts original UCI column names or standardized pythonic column names.
    """
    data = df.copy()

    # Normalize column names if raw UCI names are passed
    rename_map = {
        "Type": "type",
        "Air temperature [K]": "air_temp",
        "Process temperature [K]": "process_temp",
        "Rotational speed [rpm]": "rpm",
        "Torque [Nm]": "torque",
        "Tool wear [min]": "tool_wear",
    }
    for old_col, new_col in rename_map.items():
        if old_col in data.columns and new_col not in data.columns:
            data[new_col] = data[old_col]

    # Domain-specific feature engineering
    # 1. Temperature difference (Process temp - Air temp)
    data["temp_diff"] = data["process_temp"] - data["air_temp"]

    # 2. Mechanical power in Watts: P = Torque (Nm) * Angular velocity (rad/s)
    # Angular velocity = rpm * 2 * pi / 60
    data["power_w"] = data["torque"] * data["rpm"] * (2 * math.pi / 60.0)

    # 3. Cumulative wear-torque load product
    data["wear_torque"] = data["tool_wear"] * data["torque"]

    # One-hot encoding for machine Type (L, M, H)
    data["type_L"] = (data["type"] == "L").astype(int)
    data["type_M"] = (data["type"] == "M").astype(int)
    data["type_H"] = (data["type"] == "H").astype(int)

    # Return only the engineered feature columns in exact order
    return data[ENGINEERED_FEATURE_NAMES]


def create_features_single(raw_dict: Dict[str, Any]) -> np.ndarray:
    """
    Transforms a single telemetry reading dictionary into a 2D numpy array (1, n_features).

    Expected keys in raw_dict (case-insensitive fallback provided):
        'type', 'air_temp', 'process_temp', 'rpm', 'torque', 'tool_wear'
    """
    # Extract raw values with sensible defaults if missing
    machine_type = str(raw_dict.get("type", raw_dict.get("Type", "M"))).upper()
    air_temp = float(raw_dict.get("air_temp", raw_dict.get("Air temperature [K]", 298.0)))
    process_temp = float(raw_dict.get("process_temp", raw_dict.get("Process temperature [K]", 308.0)))
    rpm = float(raw_dict.get("rpm", raw_dict.get("Rotational speed [rpm]", 1500.0)))
    torque = float(raw_dict.get("torque", raw_dict.get("Torque [Nm]", 40.0)))
    tool_wear = float(raw_dict.get("tool_wear", raw_dict.get("Tool wear [min]", 0.0)))

    # Engineered features
    temp_diff = process_temp - air_temp
    power_w = torque * rpm * (2 * math.pi / 60.0)
    wear_torque = tool_wear * torque

    type_L = 1 if machine_type == "L" else 0
    type_M = 1 if machine_type == "M" else 0
    type_H = 1 if machine_type == "H" else 0

    features = [
        air_temp,
        process_temp,
        rpm,
        torque,
        tool_wear,
        temp_diff,
        power_w,
        wear_torque,
        type_L,
        type_M,
        type_H,
    ]

    return pd.DataFrame([features], columns=ENGINEERED_FEATURE_NAMES)


if __name__ == "__main__":
    sample_record = {
        "type": "L",
        "air_temp": 298.1,
        "process_temp": 308.6,
        "rpm": 1551,
        "torque": 42.8,
        "tool_wear": 14,
    }
    arr = create_features_single(sample_record)
    print("Single record features shape:", arr.shape)
    print("Feature names:", ENGINEERED_FEATURE_NAMES)
    print("Feature values:", arr[0].tolist())
