"""
Telemetry Ingestion Service.

Subscribes to MQTT topic 'machine/telemetry' and writes sensor data points
and engineered fields into InfluxDB bucket 'machine_data' measurement 'telemetry'.
"""

import json
import logging
import math
import os
import sys
import time
from typing import Optional

import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (IngestService) %(message)s",
)
logger = logging.getLogger(__name__)

# Read environment variables with sensible defaults
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "my-super-secret-auth-token")
INFLUX_ORG = os.getenv("INFLUX_ORG", "digital-twin-org")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "machine_data")
INFLUX_URL = os.getenv("INFLUX_URL", "http://localhost:8086")
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))

# Initialize InfluxDB Client
influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
write_api = influx_client.write_api(write_options=SYNCHRONOUS)


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

        # Parse telemetry fields (AI4I dataset fields or fallback simulated fields)
        machine_type = str(data.get("type", "M")).upper()
        air_temp = float(data.get("air_temp", data.get("temperature", 298.0)))
        process_temp = float(data.get("process_temp", air_temp + 10.0))
        rpm = float(data.get("rpm", 1500.0))
        torque = float(data.get("torque", data.get("vibration", 40.0)))
        tool_wear = float(data.get("tool_wear", 0.0))
        label = int(data.get("label", 0))

        # Compute derived features for convenient querying in Flux
        temp_diff = process_temp - air_temp
        power_w = torque * rpm * (2 * math.pi / 60.0)
        wear_torque = tool_wear * torque

        # Construct InfluxDB Point
        point = (
            Point("telemetry")
            .tag("type", machine_type)
            .field("air_temp", air_temp)
            .field("process_temp", process_temp)
            .field("rpm", rpm)
            .field("torque", torque)
            .field("tool_wear", tool_wear)
            .field("temp_diff", temp_diff)
            .field("power_w", power_w)
            .field("wear_torque", wear_torque)
            .field("label", label)
        )

        # Backwards compatibility field aliases
        point = point.field("vibration", torque).field("temperature", process_temp)

        write_api.write(bucket=INFLUX_BUCKET, record=point)
        logger.debug(f"Stored telemetry point: Type={machine_type}, ProcessTemp={process_temp}K, RPM={rpm}, Torque={torque}Nm")

    except Exception as e:
        logger.error(f"Error processing telemetry message: {e}", exc_info=True)


def run_ingest():
    logger.info(f"Starting Ingest Service connecting to InfluxDB at {INFLUX_URL} (Org: {INFLUX_ORG}, Bucket: {INFLUX_BUCKET})")

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