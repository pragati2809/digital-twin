"""
Real-time bearing-health inference service & digital-twin state.

Subscribes to 'machine/telemetry', scores each vibration window with the trained model,
writes predictions to InfluxDB measurement 'health' and publishes alerts to 'machine/health'.

failure_prob = 1 - P(normal);  fault_type = argmax of the 4-class probabilities.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import joblib
import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import FEATURE_NAMES, features_to_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (InferenceService) %(message)s")
logger = logging.getLogger(__name__)

INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "my-super-secret-auth-token")
INFLUX_ORG = os.getenv("INFLUX_ORG", "digital-twin-org")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "machine_data")
INFLUX_URL = os.getenv("INFLUX_URL", "http://localhost:8086")
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MACHINE_ID = os.getenv("MACHINE_ID", "bearing_de_01")

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best_model.joblib"))
METADATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "metadata.json"))

STATUS_CODES = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}
# Extra features copied into the 'health' measurement so one query gives the dashboard everything.
HEALTH_ECHO_FIELDS = ["rms", "kurtosis", "crest_factor", "peak_to_peak", "env_bpfo", "env_bpfi", "env_bsf"]


class DigitalTwinState:
    """In-memory representation of the twin's operational state."""

    def __init__(self):
        self.last_updated: Optional[datetime] = None
        self.tick_count = 0
        self.latest_telemetry: Dict[str, Any] = {}
        self.latest_prediction: Dict[str, Any] = {}
        self.status = "NORMAL"
        self.status_code = 0
        self.total_alerts = 0

    def update(self, telemetry: Dict[str, Any], prediction: Dict[str, Any]):
        self.last_updated = datetime.now(timezone.utc)
        self.tick_count = int(telemetry.get("t", self.tick_count + 1))
        self.latest_telemetry = telemetry
        self.latest_prediction = prediction
        self.status = prediction["status"]
        self.status_code = STATUS_CODES[self.status]
        if self.status != "NORMAL":
            self.total_alerts += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick_count,
            "status": self.status,
            "status_code": self.status_code,
            "total_alerts": self.total_alerts,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "prediction": self.latest_prediction,
        }


class InferenceService:
    def __init__(self, model_path: str = MODEL_PATH, metadata_path: str = METADATA_PATH, connect_influx: bool = True):
        self.twin_state = DigitalTwinState()
        self.threshold = 0.5
        self.normal_max = 0.3
        self.warning_max = 0.5
        self.write_api = None
        self.load_model_artifacts(model_path, metadata_path)
        if connect_influx:
            self.setup_influxdb()

    def load_model_artifacts(self, model_path: str, metadata_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at '{model_path}'. Run `python src/train.py` first.")
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.class_names = list(bundle["class_names"])
        self.feature_names = list(bundle["feature_names"])
        if self.feature_names != FEATURE_NAMES:
            raise RuntimeError("Model feature list differs from src/features.py - retrain the model.")
        self.normal_idx = self.class_names.index("normal")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                meta = json.load(f)
            self.threshold = float(meta.get("optimal_threshold", self.threshold))
            th = meta.get("status_thresholds", {})
            self.normal_max = float(th.get("normal_max", self.threshold * 0.6))
            self.warning_max = float(th.get("warning_max", self.threshold))
        logger.info(f"Loaded '{bundle.get('model_name', 'model')}' (alarm threshold {self.threshold:.2f})")

    def setup_influxdb(self):
        try:
            self.influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
            self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
            logger.info(f"InfluxDB client ready for {INFLUX_URL}")
        except Exception as e:
            logger.error(f"Failed to initialise InfluxDB client: {e}")

    def predict(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        proba = self.pipeline.predict_proba(features_to_frame(telemetry))[0]
        p_fault = float(1.0 - proba[self.normal_idx])
        cls_idx = int(proba.argmax())
        if p_fault < self.normal_max:
            status = "NORMAL"
        elif p_fault < self.warning_max:
            status = "WARNING"
        else:
            status = "CRITICAL"
        return {
            "failure_prob": round(p_fault, 4),
            "anomaly_score": round(p_fault, 4),
            "predicted_failure": int(p_fault >= self.threshold),
            "pred_class": cls_idx,
            "fault_type": self.class_names[cls_idx],
            "class_probs": {n: round(float(p), 4) for n, p in zip(self.class_names, proba)},
            "status": status,
            "threshold": self.threshold,
        }

    def process_telemetry(self, mqtt_client, payload_bytes: bytes):
        try:
            telemetry = json.loads(payload_bytes.decode("utf-8"))
            pred = self.predict(telemetry)
            self.twin_state.update(telemetry, pred)

            if self.write_api:
                point = (
                    Point("health")
                    .tag("machine", MACHINE_ID)
                    .tag("status", pred["status"])
                    .tag("fault_type", pred["fault_type"])
                    .field("failure_prob", pred["failure_prob"])
                    .field("anomaly_score", pred["anomaly_score"])
                    .field("predicted_failure", pred["predicted_failure"])
                    .field("status_code", self.twin_state.status_code)
                    .field("pred_class", pred["pred_class"])
                    .field("label", int(telemetry.get("label", 0)))
                    .field("true_class", int(telemetry.get("true_class", 0)))
                    .field("rpm", float(telemetry.get("rpm", 0.0)))
                )
                for name, p in pred["class_probs"].items():
                    point = point.field(f"p_{name}", float(p))
                for name in HEALTH_ECHO_FIELDS:
                    point = point.field(name, float(telemetry[name]))
                try:
                    self.write_api.write(bucket=INFLUX_BUCKET, record=point)
                except Exception as e:
                    logger.error(f"InfluxDB write failed: {e}")

            mqtt_client.publish("machine/health", json.dumps({
                "t": self.twin_state.tick_count,
                "failure_prob": pred["failure_prob"],
                "predicted_failure": pred["predicted_failure"],
                "fault_type": pred["fault_type"],
                "status": pred["status"],
                "status_code": self.twin_state.status_code,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }), qos=0)

            if pred["status"] != "NORMAL" or self.twin_state.tick_count % 10 == 0:
                logger.info(f"t={self.twin_state.tick_count}: {pred['status']} fault={pred['fault_type']} "
                            f"p={pred['failure_prob']:.3f} (true={telemetry.get('true_class_name', '?')})")
        except Exception as e:
            logger.error(f"Error in inference pipeline: {e}", exc_info=True)


def run_inference_service():
    service = InferenceService()
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    def on_connect(c, userdata, flags, rc, properties=None):
        if rc == 0:
            logger.info(f"Connected to MQTT Broker at {MQTT_HOST}:{MQTT_PORT}")
            c.subscribe("machine/telemetry")
        else:
            logger.error(f"Failed to connect to MQTT Broker, return code: {rc}")

    client.on_connect = on_connect
    client.on_message = lambda c, u, msg: service.process_telemetry(c, msg.payload)

    while True:
        try:
            client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
            client.loop_forever()
        except Exception as e:
            logger.warning(f"MQTT connection error: {e}. Retrying in 5 seconds...")
            time.sleep(5)


if __name__ == "__main__":
    run_inference_service()
