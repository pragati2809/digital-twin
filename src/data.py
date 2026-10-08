"""
Data loading and validation module for the AI4I 2020 Predictive Maintenance Dataset.
"""

import os
from typing import Optional
import pandas as pd

REQUIRED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

# Standardized column name map for cleaner code handling
COLUMN_RENAME_MAP = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}


def load_dataset(file_path: str = "data/ai4i2020.csv") -> pd.DataFrame:
    """
    Loads and validates the AI4I 2020 Predictive Maintenance Dataset.

    Args:
        file_path: Relative or absolute path to the CSV file.

    Returns:
        pd.DataFrame: Validated raw dataframe with original column names.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found at '{file_path}'. Please verify the file path."
        )

    df = pd.read_csv(file_path)

    # Validate required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset at '{file_path}' is missing required columns: {missing_cols}"
        )

    # Basic integrity checks
    if df.empty:
        raise ValueError(f"Dataset at '{file_path}' is empty.")

    return df


def get_standardized_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of the dataframe with standardized pythonic column names.
    """
    return df.rename(columns=COLUMN_RENAME_MAP)


if __name__ == "__main__":
    data = load_dataset()
    print(f"Dataset successfully loaded with shape: {data.shape}")
    std_data = get_standardized_dataframe(data)
    print(f"Standardized columns: {std_data.columns.tolist()}")
