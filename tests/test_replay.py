"""Tests for the replay publisher's window stream."""

from itertools import islice

from src.data import CLASS_NAMES
from src.features import FEATURE_NAMES
from src.replay_publisher import DEMO_SCENARIO, build_payload, load_window_table, window_stream


def test_demo_scenario_order_and_test_split_only():
    df = load_window_table()
    rows = list(islice(window_stream(df, "demo", "test"), sum(n for _, n in DEMO_SCENARIO)))
    assert all(r["split"] == "test" for r in rows)
    i = 0
    for cls, n in DEMO_SCENARIO:
        assert {CLASS_NAMES[int(r["class_id"])] for r in rows[i:i + n]} == {cls}
        i += n


def test_stream_loops_and_payload_schema():
    df = load_window_table()
    rows = list(islice(window_stream(df, "demo", "test"), 3 * sum(n for _, n in DEMO_SCENARIO)))
    assert len(rows) == 3 * sum(n for _, n in DEMO_SCENARIO)
    p = build_payload(rows[0], 1)
    assert set(FEATURE_NAMES) <= set(p) and {"t", "label", "true_class", "rpm", "source_file"} <= set(p)
    assert p["label"] == 0 and p["true_class_name"] == "normal"


def test_random_and_sequential_scenarios():
    df = load_window_table()
    assert len(list(islice(window_stream(df, "random", "all", seed=1), 20))) == 20
    seq = list(window_stream(df, "sequential", "test", loop=False))
    assert len(seq) == int((df["split"] == "test").sum())
