"""Records, metrics, imaging, tracking and the submission writer."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from spacepresso.core import imaging, metrics, tracking
from spacepresso.core.records import (
    TEST,
    TRAIN_ANOMALY,
    TRAIN_GOOD,
    ImageRecord,
    group_by_sample,
    parse_view,
    scan_dataset,
    select,
)


# ── records ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "filename, expected",
    [
        ("sample_view03.png", ("sample", 3)),
        ("class_01_good00_view00.png", ("class_01_good00", 0)),
        ("no_view_here.png", ("no_view_here", None)),
        ("odd_view12.jpeg", ("odd", 12)),
    ],
)
def test_parse_view(filename, expected):
    assert parse_view(filename) == expected


def test_scan_finds_all_three_splits(dataset):
    records = scan_dataset(dataset)
    assert {r.split for r in records} == {TRAIN_GOOD, TRAIN_ANOMALY, TEST}
    assert len(select(records, split=TRAIN_GOOD)) > 0


def test_only_validation_records_carry_masks(dataset):
    records = scan_dataset(dataset)
    assert all(r.has_mask for r in select(records, split=TRAIN_ANOMALY))
    assert not any(r.has_mask for r in select(records, split=TEST))


def test_scan_raises_on_a_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_dataset(tmp_path / "nope")


def test_scan_raises_when_there_are_no_classes(tmp_path):
    (tmp_path / "not_a_class").mkdir()
    with pytest.raises(FileNotFoundError, match="class_"):
        scan_dataset(tmp_path)


def test_group_by_sample_keeps_views_together(dataset):
    groups = group_by_sample(select(scan_dataset(dataset), split=TEST))
    assert all(len(g) == 2 for g in groups.values())
    for views in groups.values():
        assert [r.view for r in views] == sorted(r.view for r in views)


def test_records_are_immutable():
    """Records are shared between the runner, the detectors and the writers;
    none of them should be able to mutate another's view of the dataset."""
    record = ImageRecord(path=Path("a.png"), cls="c", split=TEST)
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.cls = "other"  # type: ignore[misc]


# ── metrics ──────────────────────────────────────────────────────────────
def test_average_precision_is_one_for_a_perfect_ranking():
    scores = np.array([0.9, 0.8, 0.1, 0.0])
    labels = np.array([1, 1, 0, 0])
    assert metrics.average_precision(scores, labels) == pytest.approx(1.0)


def test_average_precision_is_zero_without_positives():
    assert metrics.average_precision(np.array([0.5, 0.2]), np.array([0, 0])) == 0.0


def test_average_precision_matches_sklearn():
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(3)
    scores = rng.random(500)
    labels = (rng.random(500) < 0.1).astype(int)
    assert metrics.average_precision(scores, labels) == pytest.approx(
        sklearn.average_precision_score(labels, scores)
    )


def test_pooled_and_per_image_ap_differ_when_scales_differ():
    """The distinction the leaderboard punishes.

    Two images, each perfectly ranked internally, but the clean one's scores
    sit above the defective one's. Per-image AP is perfect; pooled is not.
    """
    dim = np.array([[0.1, 0.2], [0.3, 0.4]])
    bright = dim + 10.0
    masks = [np.array([[0, 0], [0, 1]]), np.array([[0, 0], [0, 0]])]

    per_image = metrics.per_image_ap(dim, masks[0])
    pooled = metrics.pooled_pixel_ap([dim, bright], masks)
    assert per_image == pytest.approx(1.0)
    assert pooled < per_image


# ── imaging ──────────────────────────────────────────────────────────────
def test_gaussian_smooth_preserves_shape_and_is_a_noop_at_zero_sigma():
    arr = np.random.default_rng(4).random((16, 20)).astype(np.float32)
    assert imaging.gaussian_smooth(arr, 2.0).shape == arr.shape
    assert np.array_equal(imaging.gaussian_smooth(arr, 0.0), arr)


def test_gaussian_smooth_reduces_variance():
    arr = np.random.default_rng(5).random((32, 32)).astype(np.float32)
    assert imaging.gaussian_smooth(arr, 2.0).var() < arr.var()


def test_resize_to_submission_hits_the_expected_shape():
    arr = np.random.default_rng(6).random((28, 28)).astype(np.float32)
    assert imaging.resize_to_submission(arr).shape == imaging.SUBMISSION_SHAPE


def test_resize_nearest_keeps_mask_values_binary():
    mask = (np.random.default_rng(7).random((16, 16)) > 0.5).astype(np.uint8)
    resized = imaging.resize_nearest(mask, (64, 64))
    assert set(np.unique(resized).tolist()) <= {0, 1}


def test_calibration_maps_into_the_unit_interval():
    maps = [np.random.default_rng(8).normal(0, 5, (8, 8)).astype(np.float32)]
    lo, hi = imaging.calibrate_to_unit(maps)
    normalised = imaging.normalise_to_unit(maps[0], lo, hi)
    assert normalised.min() >= 0.0 and normalised.max() <= 1.0


# ── tracking ─────────────────────────────────────────────────────────────
def test_digest_ignores_key_order_and_sequence_type():
    assert tracking.config_digest({"a": 1, "b": (2, 3)}) == tracking.config_digest(
        {"b": [2, 3], "a": 1}
    )


def test_digest_changes_with_the_science():
    assert tracking.config_digest({"lr": 0.1}) != tracking.config_digest({"lr": 0.2})


def test_run_id_layout_and_tag_independence():
    fingerprint = {"backbone": "resnet18"}
    with_tag = tracking.make_run_id(
        "pc", fingerprint, run_tag="exp", stamp="20260101-000000"
    )
    without = tracking.make_run_id("pc", fingerprint, stamp="20260101-000000")
    assert with_tag.startswith("20260101-000000_pc_exp_")
    # The tag labels a run; it does not change what the run computes.
    assert with_tag.split("_")[-1] == without.split("_")[-1]


def test_master_csv_widens_columns_without_shifting_old_rows(tmp_path):
    master = tmp_path / "master.csv"
    tracking.append_to_master(master, {"run_id": "a", "AP": "0.1"})
    tracking.append_to_master(
        master, {"run_id": "b", "AP": "0.2", "AP_class_09": "0.3"}
    )

    import csv

    rows = list(csv.DictReader(master.open()))
    assert [r["run_id"] for r in rows] == ["a", "b"]
    assert rows[0]["AP_class_09"] == ""
    assert rows[1]["AP_class_09"] == "0.3"
