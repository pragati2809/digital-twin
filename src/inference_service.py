"""
Real-time Machine Health Inference Service & Digital Twin State Manager.

Subscribes to 'machine/telemetry', transforms sensor features, scores failure risk
with the trained ML pipeline, writes predictions to InfluxDB measurement 'health',
and publishes health alerts to MQTT topic 'machine/health'.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import joblib
import numpy as np
import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# Add project root directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import create_features_single, ENGINEERED_FEATURE_NAMES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (InferenceService) %(message)s",
)
logger = logging.getLogger(__name__)

# Read environment variables
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "my-super-secret-auth-token")
INFLUX_ORG = os.getenv("INFLUX_ORG", "digital-twin-org")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "machine_data")
INFLUX_URL = os.getenv("INFLUX_URL", "http://localhost:8086")
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best_model.joblib"))
METADATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "metadata.json"))


class DigitalTwinState:
    """
    In-memory representation of the Digital Twin's operational state.
    """
    def __init__(self):
        self.last_updated: Optional[datetime] = None
        self.tick_count: int = 0
        self.machine_type: str = "M"
        self.latest_telemetry: Dict[str, Any] = {}
        self.latest_prediction: Dict[str, Any] = {}
        self.status: str = "NORMAL"
        self.status_code: int = 0  # 0: NORMAL, 1: WARNING, 2: CRITICAL
        self.total_alerts: int = 0

    def update(self, telemetry: Dict[str, Any], prediction: Dict[str, Any]):
        self.last_updated = datetime.now(timezone.utc)
        self.tick_count = telemetry.get("t", self.tick_count + 1)
        self.machine_type = str(telemetry.get("type", "M")).upper()
        self.latest_telemetry = telemetry
        self.latest_prediction = prediction
        self.status = prediction.get("status", "NORMAL")
        
        status_map = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}
        self.status_code = status_map.get(self.status, 0)
        
        if self.status in ["WARNING", "CRITICAL"]:
            self.total_alerts += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick_count,
            "machine_type": self.machine_type,
            "status": self.status,
            "status_code": self.status_code,
            "total_alerts": self.total_alerts,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "telemetry": self.latest_telemetry,
            "prediction": self.latest_prediction,
        }


class InferenceService:
    def __init__(self, model_path: str = MODEL_PATH, metadata_path: str = METADATA_PATH):
        self.twin_state = DigitalTwinState()
        self.model_pipeline = None
        self.metadata = {}
        self.threshold = 0.43
        self.normal_max = 0.25
        self.warning_max = 0.43

        self.load_model_artifacts(model_path, metadata_path)
        self.setup_influxdb()

    def load_model_artifacts(self, model_path: str, metadata_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model artifact not found at '{model_path}'. Run src/train.py first.")

        logger.info(f"Loading trained ML model pipeline from '{model_path}'...")
        self.model_pipeline = joblib.load(model_path)

        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
            self.threshold = float(self.metadata.get("optimal_threshold", 0.43))
            status_th = self.metadata.get("status_thresholds", {})
            self.normal_max = float(status_th.get("normal_max", self.threshold * 0.6))
            self.warning_max = float(status_th.get("warning_max", self.threshold))
            logger.info(f"Loaded metadata. Optimal Threshold={self.threshold:.4f}, NormalMax={self.normal_max:.4f}, WarningMax={self.warning_max:.4f}")
        else:
            logger.warning(f"Metadata file not found at '{metadata_path}'. Using default threshold={self.threshold}")

    def setup_influxdb(self):
        try:
            self.influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
            self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
            logger.info(f"Connected to InfluxDB at {INFLUX_URL}")
        except Exception as e:
            logger.error(f"Failed to initialize InfluxDB client: {e}")
            self.write_api = None

    def predict(self, raw_telemetry: Dict[str, Any]) -> Dict[str, Any]:
        features_arr = create_features_single(raw_telemetry)

        # Failure probability estimation
        if hasattr(self.model_pipeline, "predict_proba"):
            proba = float(self.model_pipeline.predict_proba(features_arr)[0, 1])
        else:
            proba = float(self.model_pipeline.predict(features_arr)[0])

        anomaly_score = round(proba, 4)
        predicted_failure = 1 if proba >= self.threshold else 0

        # Status categorization
        if proba < self.normal_max:
            status = "NORMAL"
        elif proba < self.warning_max:
            status = "WARNING"
        else:
            status = "CRITICAL"

        return {
            "failure_prob": round(proba, 4),
            "anomaly_score": anomaly_score,
            "predicted_failure": predicted_failure,
            "status": status,
            "threshold": self.threshold,
        }

    def process_telemetry(self, mqtt_client: mqtt.Client, payload_bytes: bytes):
        try:
            telemetry = json.loads(payload_bytes.decode("utf-8"))
            pred = self.predict(telemetry)
            self.twin_state.update(telemetry, pred)

            # Write to InfluxDB measurement 'health'
            if self.write_api:
                point = (
                    Point("health")
                    .tag("type", self.twin_state.machine_type)
                    .tag("status", pred["status"])
                    .field("failure_prob", pred["failure_prob"])
                    .field("anomaly_score", pred["anomaly_score"])
                    .field("predicted_failure", pred["predicted_failure"])
                    .field("status_code", self.twin_state.status_code)
                    .field("label", int(telemetry.get("label", 0)))
                )
                self.write_api.write(bucket=INFLUX_BUCKET, record=point)

            # Publish prediction to MQTT topic 'machine/health'
            health_payload = {
                "t": self.twin_state.tick_count,
                "type": self.twin_state.machine_type,
                "failure_prob": pred["failure_prob"],
                "predicted_failure": pred["predicted_failure"],
                "status": pred["status"],
                "status_code": self.twin_state.status_code,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            mqtt_client.publish("machine/health", json.dumps(health_payload), qos=0)

            # Log heartbeat or critical alert
            if pred["status"] in ["WARNING", "CRITICAL"] or self.twin_state.tick_count % 10 == 0:
                alert_str = " [*** ALERT CRITICAL ***]" if pred["status"] == "CRITICAL" else ""
                logger.info(f"Tick {self.twin_state.tick_count}: Status={pred['status']} (Prob={pred['failure_prob']:.4f}, Thresh={self.threshold:.2f}){alert_str}")

        except Exception as e:
            logger.error(f"Error in inference pipeline: {e}", exc_info=True)


def run_inference_service():
    service = InferenceService()
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            logger.info(f"Connected to MQTT Broker at {MQTT_HOST}:{MQTT_PORT}")
            client.subscribe("machine/telemetry")
            logger.info("Subscribed to MQTT topic 'machine/telemetry'")
        else:
            logger.error(f"Failed to connect to MQTT Broker, return code: {rc}")

    def on_message(client, userdata, msg):
        service.process_telemetry(client, msg.payload)

    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    logger.info("Starting Inference Service loop...")
    while True:
        try:
            mqtt_client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
            mqtt_client.loop_forever()
        except Exception as e:
            logger.warning(f"MQTT connection error: {e}. Retrying in 5 seconds...")
            time.sleep(5)


if __name__ == "__main__":
    run_inference_service()
