"""
Unit tests for telemetry payload parsing and point creation logic in ingest.py
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingest import on_message


class DummyMessage:
    def __init__(self, payload: bytes):
        self.payload = payload


def test_ingest_payload_parsing():
    sample_payload = {
        "t": 105,
        "udi": 12,
        "product_id": "L47181",
        "type": "L",
        "air_temp": 298.1,
        "process_temp": 308.6,
        "rpm": 1551,
        "torque": 42.8,
        "tool_wear": 14,
        "label": 1,
    }
    msg = DummyMessage(json.dumps(sample_payload).encode("utf-8"))

    # Execute on_message logic without raising exception
    try:
        on_message(None, None, msg)
        success = True
    except Exception as e:
        success = False

    assert success, "on_message must parse valid JSON telemetry payload cleanly."
