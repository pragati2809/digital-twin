"""
Dataset Replay Publisher for Digital Twin system.

Reads rows from the AI4I 2020 dataset CSV and publishes real sensor telemetry
to the MQTT broker topic 'machine/telemetry' using Paho MQTT v2 API.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Optional

# Add project root directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import paho.mqtt.client as mqtt
from src.data import load_dataset, get_standardized_dataframe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (ReplayPublisher) %(message)s",
)
logger = logging.getLogger(__name__)


def create_mqtt_client(broker_host: str, broker_port: int) -> mqtt.Client:
    """
    Creates and connects an MQTT client using Paho MQTT v2 API.
    """
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            logger.info(f"Connected to MQTT Broker at {broker_host}:{broker_port}")
        else:
            logger.error(f"Failed to connect to MQTT Broker, return code: {rc}")

    def on_disconnect(client, userdata, flags, rc, properties=None):
        logger.warning(f"Disconnected from MQTT Broker (rc={rc}). Reconnecting...")

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect

    client.connect(broker_host, broker_port, keepalive=60)
    client.loop_start()
    return client


def run_replay(
    csv_path: str = "data/ai4i2020.csv",
    broker_host: str = "localhost",
    broker_port: int = 1883,
    rate_hz: float = 1.0,
    loop: bool = True,
    start_row: int = 0,
    shuffle_seed: Optional[int] = None,
    max_rows: Optional[int] = None,
):
    df_raw = load_dataset(csv_path)
    df_std = get_standardized_dataframe(df_raw)

    if start_row > 0:
        df_std = df_std.iloc[start_row:].reset_index(drop=True)

    if shuffle_seed is not None:
        df_std = df_std.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)

    if max_rows is not None:
        df_std = df_std.head(max_rows)

    client = create_mqtt_client(broker_host, broker_port)
    delay_sec = 1.0 / max(0.1, rate_hz)

    row_index = 0
    t_tick = 0
    total_rows = len(df_std)

    logger.info(f"Starting replay stream of {total_rows} rows at {rate_hz} Hz (delay={delay_sec:.2f}s per tick)...")

    try:
        while True:
            row = df_std.iloc[row_index]
            t_tick += 1

            payload = {
                "t": t_tick,
                "udi": int(row["udi"]),
                "product_id": str(row["product_id"]),
                "type": str(row["type"]),
                "air_temp": float(row["air_temp"]),
                "process_temp": float(row["process_temp"]),
                "rpm": float(row["rpm"]),
                "torque": float(row["torque"]),
                "tool_wear": float(row["tool_wear"]),
                "label": int(row["machine_failure"]),  # Ground truth label for evaluation
            }

            json_payload = json.dumps(payload)
            client.publish("machine/telemetry", json_payload, qos=0)

            if t_tick % 10 == 0 or row["machine_failure"] == 1:
                fail_flag = " [FAILURE ROW]" if row["machine_failure"] == 1 else ""
                logger.info(f"Tick {t_tick} (Row {row_index+1}/{total_rows}): Type={payload['type']} Temp={payload['process_temp']}K RPM={payload['rpm']} Torque={payload['torque']}Nm Wear={payload['tool_wear']}min{fail_flag}")

            row_index += 1
            if row_index >= total_rows:
                if loop:
                    logger.info("Reached end of dataset; looping back to row 0.")
                    row_index = 0
                else:
                    logger.info("Reached end of dataset; replay finished.")
                    break

            time.sleep(delay_sec)

    except KeyboardInterrupt:
        logger.info("Replay publisher stopped by user.")
    finally:
        client.loop_stop()
        client.disconnect()
        logger.info("MQTT client disconnected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI4I 2020 Telemetry Replay Publisher")
    parser.add_argument("--csv", type=str, default="data/ai4i2020.csv", help="Path to CSV file")
    parser.add_argument("--host", type=str, default=os.getenv("MQTT_HOST", "localhost"), help="MQTT broker host")
    parser.add_argument("--port", type=int, default=int(os.getenv("MQTT_PORT", 1883)), help="MQTT broker port")
    parser.add_argument("--rate", type=float, default=1.0, help="Publish rate in Hz")
    parser.add_argument("--no-loop", action="store_true", help="Disable looping when reaching end of dataset")
    parser.add_argument("--start-row", type=int, default=0, help="Starting row index")
    parser.add_argument("--seed", type=int, default=None, help="Shuffle random seed")
    parser.add_argument("--max-rows", type=int, default=None, help="Maximum number of rows to replay")

    args = parser.parse_args()

    run_replay(
        csv_path=args.csv,
        broker_host=args.host,
        broker_port=args.port,
        rate_hz=args.rate,
        loop=not args.no_loop,
        start_row=args.start_row,
        shuffle_seed=args.seed,
        max_rows=args.max_rows,
    )
