"""Tests for the CWRU loader and the leak-aware split."""

import numpy as np
import pytest

from src.data import (CLASS_NAMES, RECORDING_REGISTRY, WINDOW_SIZE, WINDOW_STRIDE, assign_splits,
                      build_window_table, find_de_key, find_rpm, load_all_recordings, load_recording,
                      window_starts)
from src.features import FEATURE_NAMES, TARGET_FS


def test_find_de_key_uses_recording_id_not_first_key():
    # Mimics 99.mat: contains another recording's channel (X098) as well as its own.
    mat = {"ans": 1, "X098_DE_time": 0, "X099_DE_time": 1, "X099_FE_time": 2}
    assert find_de_key(mat, 99) == "X099_DE_time"
    with pytest.raises(KeyError):
        find_de_key({"X098_DE_time": 0, "X097_DE_time": 1}, 99)


def test_find_rpm_falls_back_to_default():
    assert find_rpm({"X097RPM": [[1796]]}, 97, 1797) == (1796.0, "file")
    assert find_rpm({}, 98, 1772) == (1772.0, "default")  # 98.mat has no RPM variable
    assert find_rpm({"X098RPM": [[5]]}, 98, 1772)[1] == "default"  # implausible value ignored


def test_real_files_load_and_resample_to_12k():
    recs = load_all_recordings()
    assert {97, 98, 99, 100, 105, 118, 130} <= set(recs)
    for r in recs.values():
        assert r.signal.ndim == 1 and np.all(np.isfinite(r.signal))
        assert 4.5 < r.duration_s < 11  # 12 kHz-equivalent length (normal files were 48 kHz)
        assert r.de_key == f"X{r.file_id:03d}_DE_time"
    assert recs[99].de_key == "X099_DE_time"  # not the stray X098 arrays inside 99.mat
    assert recs[98].rpm_source == "default"


def test_registry_classes():
    assert RECORDING_REGISTRY[105][0] == "inner_race"
    assert RECORDING_REGISTRY[118][0] == "ball"
    assert RECORDING_REGISTRY[130][0] == "outer_race"
    with pytest.raises(KeyError):
        load_recording(9999)


@pytest.fixture(scope="module")
def table():
    return assign_splits(build_window_table())


def test_table_columns_and_labels(table):
    assert all(c in table.columns for c in FEATURE_NAMES + ["split", "class_id", "label", "source_file"])
    assert set(table["class_id"].unique()) == {0, 1, 2, 3}
    assert (table["label"] == (table["class_id"] != 0).astype(int)).all()
    assert np.all(np.isfinite(table[FEATURE_NAMES].to_numpy()))


def test_every_split_has_every_class(table):
    for s in ("train", "val", "test"):
        assert set(table.loc[table["split"] == s, "class_id"]) == {0, 1, 2, 3}, s


def test_no_window_overlap_between_splits(table):
    """Windows from the same recording in different splits must not share any samples."""
    for fid, g in table[table["split"] != "drop"].groupby("source_file"):
        spans = {}
        for s, gs in g.groupby("split"):
            starts = gs["win_idx"].to_numpy() * WINDOW_STRIDE
            spans[s] = (starts.min(), starts.max() + WINDOW_SIZE)
        names = sorted(spans, key=lambda k: spans[k][0])
        for a, b in zip(names, names[1:]):
            assert spans[a][1] <= spans[b][0], f"file {fid}: {a} overlaps {b}"


def test_normal_test_is_unseen_recording(table):
    normal = table[table["class_id"] == 0]
    test_files = set(normal.loc[normal["split"] == "test", "source_file"])
    train_files = set(normal.loc[normal["split"] == "train", "source_file"])
    assert test_files and not (test_files & train_files)


def test_window_starts():
    s = window_starts(WINDOW_SIZE + 3 * WINDOW_STRIDE)
    assert len(s) == 4 and s[-1] + WINDOW_SIZE == WINDOW_SIZE + 3 * WINDOW_STRIDE
