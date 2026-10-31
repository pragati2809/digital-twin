"""
Telemetry Ingestion Service.

Subscribes to MQTT topic 'machine/telemetry' (one message = one vibration window summarised
as 18 features, see src/features.py) and writes it to InfluxDB bucket 'machine_data',
measurement 'telemetry'.
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

from src.features import FEATURE_NAMES, features_to_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (IngestService) %(message)s")
logger = logging.getLogger(__name__)

INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "my-super-secret-auth-token")
INFLUX_ORG = os.getenv("INFLUX_ORG", "digital-twin-org")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "machine_data")
INFLUX_URL = os.getenv("INFLUX_URL", "http://localhost:8086")
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MACHINE_ID = os.getenv("MACHINE_ID", "bearing_de_01")

_influx_client: Optional[InfluxDBClient] = None
_write_api = None


def get_write_api():
    global _influx_client, _write_api
    if _write_api is None:
        _influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        _write_api = _influx_client.write_api(write_options=SYNCHRONOUS)
    return _write_api


def build_point(data: Dict[str, Any]) -> Point:
    """Validate a telemetry payload and convert it to an InfluxDB point (raises ValueError)."""
    features_to_frame(data)  # raises if a feature is missing or not finite
    point = (
        Point("telemetry")
        .tag("machine", MACHINE_ID)
        .field("rpm", float(data.get("rpm", 0.0)))
        .field("label", int(data.get("label", 0)))
        .field("true_class", int(data.get("true_class", 0)))
        .field("source_file", int(data.get("source_file", 0)))
    )
    for name in FEATURE_NAMES:
        point = point.field(name, float(data[name]))
    return point


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        logger.info(f"Connected to MQTT Broker at {MQTT_HOST}:{MQTT_PORT}")
        client.subscribe("machine/telemetry")
        logger.info("Subscribed to MQTT topic 'machine/telemetry'")
    else:
        logger.error(f"Failed to connect to MQTT Broker, return code: {rc}")


def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode("utf-8"))
        get_write_api().write(bucket=INFLUX_BUCKET, record=build_point(data))
        logger.debug(f"Stored window t={data.get('t')} rms={data.get('rms')}")
    except Exception as e:
        logger.error(f"Error processing telemetry message: {e}")


def run_ingest():
    logger.info(f"Starting Ingest Service -> InfluxDB {INFLUX_URL} (org={INFLUX_ORG}, bucket={INFLUX_BUCKET})")
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    while True:
        try:
            mqtt_client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
            mqtt_client.loop_forever()
        except Exception as e:
            logger.warning(f"MQTT connection error: {e}. Retrying in 5 seconds...")
            time.sleep(5)


if __name__ == "__main__":
    run_ingest()
