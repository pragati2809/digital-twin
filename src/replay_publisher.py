"""
CWRU replay publisher.

Streams vibration-window features (computed from the real CWRU recordings) to MQTT topic
'machine/telemetry'. By default it replays only windows from the held-out TEST split, in a
demo scenario:  normal -> inner race -> normal -> ball -> normal -> outer race -> (repeat)

Each message stands for one 2048-sample window (0.17 s of data); at 1 Hz the replay is
therefore accelerated ~6x relative to real time.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, Iterator, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt

from src.data import CLASS_NAMES, DEFAULT_DATA_DIR, assign_splits, build_window_table
from src.features import FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (ReplayPublisher) %(message)s")
logger = logging.getLogger(__name__)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WINDOW_CSV = os.path.join(ROOT, "data", "cwru_windows.csv")

DEMO_SCENARIO = [("normal", 40), ("inner_race", 25), ("normal", 25), ("ball", 25),
                 ("normal", 25), ("outer_race", 25)]


def load_window_table(csv_path: str = WINDOW_CSV) -> pd.DataFrame:
    """Windows with features + split column (cached CSV written by train.py, else rebuilt)."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if all(c in df.columns for c in FEATURE_NAMES + ["split", "class_id"]):
            return df
        logger.warning("Cached window table is outdated; rebuilding from .mat files")
    return assign_splits(build_window_table(data_dir=DEFAULT_DATA_DIR))


def build_payload(row: pd.Series, tick: int) -> Dict:
    payload = {
        "t": tick,
        "source_file": int(row["source_file"]),
        "win_idx": int(row["win_idx"]),
        "rpm": float(row["rpm"]),
        "true_class": int(row["class_id"]),
        "true_class_name": CLASS_NAMES[int(row["class_id"])],
        "label": int(row["class_id"] != 0),
    }
    payload.update({k: round(float(row[k]), 6) for k in FEATURE_NAMES})
    return payload


def window_stream(df: pd.DataFrame, scenario: str = "demo", split: str = "test",
                  seed: int = 0, loop: bool = True) -> Iterator[pd.Series]:
    """Yield rows (windows) according to the chosen scenario."""
    pool = df if split == "all" else df[df["split"] == split]
    pool = pool.sort_values(["source_file", "win_idx"]).reset_index(drop=True)
    if pool.empty:
        raise ValueError(f"No windows in split '{split}'")
    by_class = {c: pool[pool["class_id"] == i].reset_index(drop=True) for i, c in enumerate(CLASS_NAMES)}
    missing = [c for c, d in by_class.items() if d.empty]
    if missing and scenario == "demo":
        raise ValueError(f"No windows for classes {missing} in split '{split}'")

    if scenario == "sequential":
        while True:
            for _, row in pool.iterrows():
                yield row
            if not loop:
                return
    elif scenario == "random":
        rng = np.random.default_rng(seed)
        while True:
            yield pool.iloc[int(rng.integers(len(pool)))]
    else:  # demo
        pointers = {c: 0 for c in CLASS_NAMES}
        while True:
            for cls, n in DEMO_SCENARIO:
                d = by_class[cls]
                for _ in range(n):
                    yield d.iloc[pointers[cls] % len(d)]
                    pointers[cls] += 1
            if not loop:
                return


def create_mqtt_client(host: str, port: int, retries: int = 30) -> mqtt.Client:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = lambda c, u, f, rc, p=None: logger.info(f"Connected to MQTT {host}:{port} (rc={rc})")
    client.on_disconnect = lambda c, u, f, rc, p=None: logger.warning(f"Disconnected from MQTT (rc={rc})")
    for attempt in range(1, retries + 1):
        try:
            client.connect(host, port, keepalive=60)
            break
        except OSError as e:
            logger.warning(f"MQTT broker not reachable ({e}); retry {attempt}/{retries} in 3 s")
            time.sleep(3)
    else:
        raise ConnectionError(f"Could not connect to MQTT broker at {host}:{port}")
    client.loop_start()
    return client


def run_replay(host: str, port: int, rate_hz: float, scenario: str, split: str,
               loop: bool, max_ticks: Optional[int], seed: int):
    df = load_window_table()
    client = create_mqtt_client(host, port)
    delay = 1.0 / max(0.1, rate_hz)
    logger.info(f"Replaying CWRU windows (scenario={scenario}, split={split}) at {rate_hz} Hz")
    tick = 0
    try:
        for row in window_stream(df, scenario, split, seed, loop):
            tick += 1
            payload = build_payload(row, tick)
            client.publish("machine/telemetry", json.dumps(payload), qos=0)
            if tick % 10 == 0 or payload["label"]:
                logger.info(f"t={tick} file={payload['source_file']} state={payload['true_class_name']:<10} "
                            f"rms={payload['rms']:.3f} kurt={payload['kurtosis']:.2f}")
            if max_ticks and tick >= max_ticks:
                break
            time.sleep(delay)
    except KeyboardInterrupt:
        logger.info("Replay stopped by user.")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CWRU vibration replay publisher")
    p.add_argument("--host", default=os.getenv("MQTT_HOST", "localhost"))
    p.add_argument("--port", type=int, default=int(os.getenv("MQTT_PORT", 1883)))
    p.add_argument("--rate", type=float, default=1.0, help="windows per second")
    p.add_argument("--scenario", choices=["demo", "sequential", "random"], default="demo")
    p.add_argument("--split", choices=["test", "val", "train", "all"], default="test")
    p.add_argument("--no-loop", action="store_true")
    p.add_argument("--loop", action="store_true", help="(default) loop forever; kept for compatibility")
    p.add_argument("--max-ticks", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    run_replay(a.host, a.port, a.rate, a.scenario, a.split, not a.no_loop, a.max_ticks, a.seed)
