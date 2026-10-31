"""Tests for ingest.py payload handling."""

import json

import pytest

import ingest
from src.features import FEATURE_NAMES
from src.replay_publisher import build_payload, load_window_table


@pytest.fixture(scope="module")
def payload():
    df = load_window_table()
    return build_payload(df[df["class_id"] == 2].iloc[3], 5)


class Msg:
    def __init__(self, payload: bytes):
        self.payload = payload


class FakeWriter:
    def __init__(self):
        self.records = []

    def write(self, bucket, record):
        self.records.append((bucket, record))


def test_build_point_contains_all_features(payload):
    line = ingest.build_point(payload).to_line_protocol()
    assert line.startswith("telemetry,machine=")
    for name in FEATURE_NAMES:
        assert f"{name}=" in line
    assert "label=1i" in line and "true_class=2i" in line


def test_build_point_rejects_incomplete_payload(payload):
    bad = {k: v for k, v in payload.items() if k != "kurtosis"}
    with pytest.raises(ValueError):
        ingest.build_point(bad)


def test_on_message_writes_valid_and_survives_invalid(payload, monkeypatch):
    writer = FakeWriter()
    monkeypatch.setattr(ingest, "get_write_api", lambda: writer)
    ingest.on_message(None, None, Msg(json.dumps(payload).encode()))
    ingest.on_message(None, None, Msg(b"garbage"))
    ingest.on_message(None, None, Msg(b'{"rms": 1}'))
    assert len(writer.records) == 1 and writer.records[0][0] == ingest.INFLUX_BUCKET
