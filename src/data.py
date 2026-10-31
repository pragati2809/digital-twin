"""
CWRU bearing dataset loader.

Reads Case Western Reserve University bearing vibration recordings (.mat), finds the
correct drive-end (DE) accelerometer channel inside each file, resamples everything to
a common 12 kHz rate, slices it into overlapping windows, extracts vibration features
and assigns leak-aware train / validation / test splits.

Known quirks of the CWRU files handled here:
  * Normal-baseline files (97-100) were recorded at 48 kHz, fault files at 12 kHz.
  * 99.mat also contains X098_* arrays and an `ans` variable.
  * 98.mat has no RPM variable.
  => the loader looks keys up by the recording ID, never by blindly taking "the" DE key.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy.io import loadmat

from src.features import FEATURE_NAMES, TARGET_FS, extract_features

DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cwru"))

WINDOW_SIZE = 2048  # samples @ 12 kHz  (~0.17 s)
WINDOW_STRIDE = 512  # 75 % overlap

CLASS_NAMES = ["normal", "inner_race", "ball", "outer_race"]
CLASS_LABELS = {  # human readable
    "normal": "Normal",
    "inner_race": "Inner Race Fault",
    "ball": "Ball Fault",
    "outer_race": "Outer Race Fault",
}

# id -> (class_name, original_fs, load_hp, fault_size_inch, default_rpm)
# The first 7 are the files shipped in data/cwru/. The rest are optional extras from the
# CWRU 12 kHz drive-end tables: if you download them into data/cwru/ they are picked up
# automatically (verify the IDs against the CWRU website; files that are absent are skipped).
RECORDING_REGISTRY: Dict[int, tuple] = {
    # --- normal baseline (48 kHz) ---
    97: ("normal", 48000, 0, 0.0, 1797),
    98: ("normal", 48000, 1, 0.0, 1772),
    99: ("normal", 48000, 2, 0.0, 1750),
    100: ("normal", 48000, 3, 0.0, 1730),
    # --- 0.007" faults @ 0 HP (12 kHz) ---
    105: ("inner_race", 12000, 0, 0.007, 1797),
    118: ("ball", 12000, 0, 0.007, 1797),
    130: ("outer_race", 12000, 0, 0.007, 1797),
    # --- optional extras: 0.007" @ 1-3 HP ---
    106: ("inner_race", 12000, 1, 0.007, 1772),
    107: ("inner_race", 12000, 2, 0.007, 1750),
    108: ("inner_race", 12000, 3, 0.007, 1730),
    119: ("ball", 12000, 1, 0.007, 1772),
    120: ("ball", 12000, 2, 0.007, 1750),
    121: ("ball", 12000, 3, 0.007, 1730),
    131: ("outer_race", 12000, 1, 0.007, 1772),
    132: ("outer_race", 12000, 2, 0.007, 1750),
    133: ("outer_race", 12000, 3, 0.007, 1730),
    # --- optional extras: 0.014" @ 0-3 HP ---
    169: ("inner_race", 12000, 0, 0.014, 1797),
    170: ("inner_race", 12000, 1, 0.014, 1772),
    171: ("inner_race", 12000, 2, 0.014, 1750),
    172: ("inner_race", 12000, 3, 0.014, 1730),
    185: ("ball", 12000, 0, 0.014, 1797),
    186: ("ball", 12000, 1, 0.014, 1772),
    187: ("ball", 12000, 2, 0.014, 1750),
    188: ("ball", 12000, 3, 0.014, 1730),
    197: ("outer_race", 12000, 0, 0.014, 1797),
    198: ("outer_race", 12000, 1, 0.014, 1772),
    199: ("outer_race", 12000, 2, 0.014, 1750),
    200: ("outer_race", 12000, 3, 0.014, 1730),
}


@dataclass
class Recording:
    file_id: int
    class_name: str
    class_id: int
    load_hp: int
    fault_size: float
    rpm: float
    original_fs: int
    signal: np.ndarray  # resampled to TARGET_FS, float64
    de_key: str
    rpm_source: str  # "file" or "default"

    @property
    def n_samples(self) -> int:
        return int(len(self.signal))

    @property
    def duration_s(self) -> float:
        return self.n_samples / TARGET_FS


def find_de_key(mat: dict, file_id: int) -> str:
    """Return the drive-end time-series key that belongs to `file_id`."""
    expected = f"X{file_id:03d}_DE_time"
    if expected in mat:
        return expected
    # Tolerate different zero padding (X97_DE_time) but never use another recording's key.
    for key in mat:
        m = re.fullmatch(r"X(\d+)_DE_time", key)
        if m and int(m.group(1)) == file_id:
            return key
    candidates = [k for k in mat if k.endswith("_DE_time")]
    if len(candidates) == 1:  # file with a single DE channel whatever it is called
        return candidates[0]
    raise KeyError(f"No DE_time channel for recording {file_id} (keys: {sorted(candidates)})")


def find_rpm(mat: dict, file_id: int, default: float):
    for key in (f"X{file_id:03d}RPM", f"X{file_id}RPM"):
        if key in mat:
            try:
                val = float(np.ravel(mat[key])[0])
                if 500 < val < 4000:
                    return val, "file"
            except (TypeError, ValueError, IndexError):
                pass
    return float(default), "default"


def load_recording(file_id: int, data_dir: str = DEFAULT_DATA_DIR) -> Recording:
    if file_id not in RECORDING_REGISTRY:
        raise KeyError(f"Recording {file_id} is not in RECORDING_REGISTRY")
    path = os.path.join(data_dir, f"{file_id}.mat")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CWRU file not found: {path}")

    class_name, fs, load_hp, fault_size, default_rpm = RECORDING_REGISTRY[file_id]
    mat = loadmat(path)
    key = find_de_key(mat, file_id)
    x = np.asarray(mat[key], dtype=np.float64).ravel()
    if x.size < WINDOW_SIZE * 4 or not np.all(np.isfinite(x)):
        raise ValueError(f"Recording {file_id}: signal too short or contains NaN/inf")
    rpm, rpm_src = find_rpm(mat, file_id, default_rpm)

    if fs != TARGET_FS:
        if fs % TARGET_FS != 0:
            raise ValueError(f"Cannot decimate {fs} Hz to {TARGET_FS} Hz")
        x = sps.resample_poly(x, up=1, down=fs // TARGET_FS)  # anti-aliased

    return Recording(
        file_id=file_id,
        class_name=class_name,
        class_id=CLASS_NAMES.index(class_name),
        load_hp=load_hp,
        fault_size=fault_size,
        rpm=rpm,
        original_fs=fs,
        signal=x,
        de_key=key,
        rpm_source=rpm_src,
    )


def available_recording_ids(data_dir: str = DEFAULT_DATA_DIR) -> List[int]:
    ids = []
    if os.path.isdir(data_dir):
        for name in os.listdir(data_dir):
            m = re.fullmatch(r"(\d+)\.mat", name)
            if m and int(m.group(1)) in RECORDING_REGISTRY:
                ids.append(int(m.group(1)))
    return sorted(ids)


def load_all_recordings(data_dir: str = DEFAULT_DATA_DIR) -> Dict[int, Recording]:
    ids = available_recording_ids(data_dir)
    if not ids:
        raise FileNotFoundError(f"No CWRU .mat files found in '{data_dir}'")
    return {i: load_recording(i, data_dir) for i in ids}


def recording_summary(recordings: Dict[int, Recording]) -> pd.DataFrame:
    rows = [
        {
            "file": f"{r.file_id}.mat",
            "class": CLASS_LABELS[r.class_name],
            "load_hp": r.load_hp,
            "fault_size_in": r.fault_size,
            "rpm": r.rpm,
            "rpm_source": r.rpm_source,
            "original_fs_hz": r.original_fs,
            "samples_12k": r.n_samples,
            "duration_s": round(r.duration_s, 2),
            "channel": r.de_key,
        }
        for r in recordings.values()
    ]
    return pd.DataFrame(rows)


def window_starts(n_samples: int, win: int = WINDOW_SIZE, stride: int = WINDOW_STRIDE) -> np.ndarray:
    return np.arange(0, n_samples - win + 1, stride)


def get_window(rec: Recording, win_idx: int, win: int = WINDOW_SIZE, stride: int = WINDOW_STRIDE) -> np.ndarray:
    start = int(win_idx) * stride
    return rec.signal[start : start + win]


def build_window_table(recordings: Optional[Dict[int, Recording]] = None,
                       data_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """One row per window: metadata + FEATURE_NAMES columns."""
    if recordings is None:
        recordings = load_all_recordings(data_dir)
    rows = []
    for fid, rec in recordings.items():
        for wi, start in enumerate(window_starts(rec.n_samples)):
            feats = extract_features(rec.signal[start : start + WINDOW_SIZE], fs=TARGET_FS, rpm=rec.rpm)
            rows.append({
                "source_file": fid,
                "win_idx": wi,
                "class_id": rec.class_id,
                "class_name": rec.class_name,
                "label": int(rec.class_id != 0),
                "load_hp": rec.load_hp,
                "rpm": rec.rpm,
                **feats,
            })
    df = pd.DataFrame(rows)
    return df[["source_file", "win_idx", "class_id", "class_name", "label", "load_hp", "rpm"] + FEATURE_NAMES]


def assign_splits(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2) -> pd.DataFrame:
    """
    Leak-aware split. Windows overlap by 75 %, so a random split would put near-duplicate
    windows in train and test. Instead:

      * classes with >= 3 recordings  -> recording-level split
            (last recording = test, second to last = validation, rest = train)
      * classes with < 3 recordings   -> contiguous time blocks per recording
            (first 60 % train | gap | next 20 % val | gap | last 20 % test), with a purge
            gap wider than one window so no sample is shared between splits.
    """
    gap = int(np.ceil(WINDOW_SIZE / WINDOW_STRIDE))  # windows that overlap a given window
    out = df.copy()
    out["split"] = "drop"

    for _, cdf in out.groupby("class_id"):
        files = sorted(cdf["source_file"].unique())
        if len(files) >= 3:
            out.loc[out["source_file"] == files[-1], "split"] = "test"
            out.loc[out["source_file"] == files[-2], "split"] = "val"
            out.loc[out["source_file"].isin(files[:-2]), "split"] = "train"
        else:
            for f in files:
                idx = out.index[out["source_file"] == f]
                n = len(idx)
                n_tr = int(n * train_frac)
                n_va = int(n * val_frac)
                out.loc[idx[:n_tr], "split"] = "train"
                out.loc[idx[n_tr + gap : n_tr + gap + n_va], "split"] = "val"
                out.loc[idx[n_tr + gap + n_va + gap :], "split"] = "test"
    return out


def describe_split_protocol(df: pd.DataFrame) -> str:
    parts = []
    for cid, cdf in df.groupby("class_id"):
        n_files = cdf["source_file"].nunique()
        mode = "recording-level" if n_files >= 3 else "time-block (purged)"
        parts.append(f"{CLASS_NAMES[cid]}: {n_files} recording(s) -> {mode}")
    return "; ".join(parts)


if __name__ == "__main__":
    recs = load_all_recordings()
    print(recording_summary(recs).to_string(index=False))
