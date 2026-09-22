"""Rank normalisation, features, calibration, cross-validation and fusion."""

from __future__ import annotations

import numpy as np
import pytest

from spacepresso.stacking import fusion
from spacepresso.stacking.calibration import fit_calibrator
from spacepresso.stacking.crossval import ClassTrainingData, out_of_fold
from spacepresso.stacking.features import FeatureConfig, feature_names, featurize_image
from spacepresso.stacking.normalisation import rank_normalise, rank_transform


# ── normalisation ────────────────────────────────────────────────────────
def test_rank_transform_is_monotone_and_spans_the_unit_interval():
    values = np.array([5.0, 1.0, 3.0, 9.0])
    ranks = rank_transform(values)
    assert np.argsort(ranks).tolist() == np.argsort(values).tolist()
    assert ranks.min() == 0.0 and ranks.max() == 1.0


def test_rank_transform_breaks_ties_stably():
    """These maps have large constant regions; a random tie-break would
    inject noise exactly where the metric is most sensitive."""
    values = np.array([2.0, 2.0, 2.0, 1.0])
    first = rank_transform(values)
    assert np.array_equal(first, rank_transform(values))
    assert first[3] == 0.0


def test_global_scope_preserves_between_image_ordering():
    scores = np.zeros((2, 4, 4, 1), dtype=np.float32)
    scores[0] = 0.1
    scores[1] = 10.0
    out = rank_normalise(scores, "global")
    assert out[1].mean() > out[0].mean()


def test_per_class_scope_removes_between_class_offsets():
    scores = np.zeros((2, 4, 4, 1), dtype=np.float32)
    scores[0] = 0.1
    scores[1] = 10.0
    out = rank_normalise(scores, "per_class", classes=["a", "b"])
    assert out[0].mean() == pytest.approx(out[1].mean())


def test_methods_are_normalised_independently():
    scores = np.stack(
        [np.full((3, 4, 4), 0.1), np.full((3, 4, 4), 500.0)], axis=-1
    ).astype(np.float32)
    out = rank_normalise(scores, "global")
    assert out[..., 0].mean() == pytest.approx(out[..., 1].mean())


def test_inplace_actually_writes_through():
    scores = np.random.default_rng(0).random((2, 4, 4, 2)).astype(np.float32)
    returned = rank_normalise(scores, "global", inplace=True)
    assert returned is scores


def test_none_scope_is_a_noop():
    scores = np.random.default_rng(1).random((2, 4, 4, 1)).astype(np.float32)
    assert np.array_equal(rank_normalise(scores.copy(), "none"), scores)


# ── features ─────────────────────────────────────────────────────────────
def test_names_and_columns_cannot_drift():
    maps = [
        np.random.default_rng(2).random((8, 8)).astype(np.float32) for _ in range(3)
    ]
    matrix, names = featurize_image(
        maps,
        FeatureConfig(),
        class_id="class_01",
        all_classes=["class_01", "class_02"],
        view=0,
    )
    assert matrix.shape == (64, len(names))
    assert len(names) == len(set(names)), "duplicate feature names"


def test_feature_names_matches_a_real_featurisation():
    config = FeatureConfig(spatial_prior=False)
    maps = [np.zeros((8, 8), dtype=np.float32) for _ in range(2)]
    _, actual = featurize_image(
        maps, config, class_id="a", all_classes=["a", "b"], view=0
    )
    assert feature_names(config, 2, all_classes=["a", "b"]) == actual


def test_cross_method_features_vanish_with_a_single_method():
    config = FeatureConfig()
    one = featurize_image([np.zeros((8, 8), np.float32)], config)[1]
    two = featurize_image([np.zeros((8, 8), np.float32)] * 2, config)[1]
    assert not [n for n in one if n.startswith(("x_", "xc_"))]
    assert [n for n in two if n.startswith(("x_", "xc_"))]


def test_min_top_k_requires_agreement():
    """High only where at least k methods agree, unlike the mean."""
    shape = (4, 4)
    loud = np.ones(shape, dtype=np.float32)
    quiet = np.zeros(shape, dtype=np.float32)
    config = FeatureConfig(
        raw_score=False,
        per_image_rank=False,
        gaussian_sigmas=(),
        window_sizes=(),
        image_aggregates=(),
        cross_stats=False,
        cross_consensus=False,
        cross_cv=False,
        min_top_k=3,
        spatial_coords=False,
        spatial_prior=False,
        class_onehot=False,
        view_onehot=False,
    )
    matrix, names = featurize_image([loud, loud, quiet], config)
    assert names == ["xc_min_top3"]
    assert matrix.max() == 0.0  # the third method disagrees


def test_unknown_image_aggregate_is_rejected():
    with pytest.raises(ValueError, match="unknown image aggregate"):
        featurize_image(
            [np.zeros((4, 4), np.float32)], FeatureConfig(image_aggregates=("nope",))
        )


# ── calibration ──────────────────────────────────────────────────────────
def test_isotonic_calibration_is_monotone():
    rng = np.random.default_rng(3)
    predictions = rng.random(5000)
    labels = (rng.random(5000) < predictions * 0.5).astype(int)
    calibrator = fit_calibrator("isotonic", predictions, labels)
    probe = np.linspace(0, 1, 50)
    calibrated = calibrator.apply(probe)
    assert np.all(np.diff(calibrated) >= -1e-6)


def test_calibration_preserves_ranking():
    rng = np.random.default_rng(4)
    predictions = rng.random(2000)
    labels = (rng.random(2000) < predictions).astype(int)
    calibrator = fit_calibrator("isotonic", predictions, labels)
    out = calibrator.apply(predictions)
    # Monotone maps cannot invert any pair.
    order_before = np.argsort(predictions, kind="stable")
    assert np.all(np.diff(out[order_before]) >= -1e-6)


def test_single_class_labels_disable_calibration_rather_than_crashing():
    calibrator = fit_calibrator("isotonic", np.random.rand(100), np.zeros(100, int))
    assert calibrator.method == "none"


def test_stratified_downsample_keeps_positives():
    rng = np.random.default_rng(5)
    n = 200_000
    predictions = rng.random(n)
    labels = np.zeros(n, dtype=int)
    labels[:400] = 1  # 0.2% positive
    calibrator = fit_calibrator("isotonic", predictions, labels, max_rows=10_000)
    assert calibrator.method == "isotonic"


def test_none_method_is_the_identity():
    values = np.array([0.1, 0.9])
    assert np.allclose(
        fit_calibrator("none", values, np.array([0, 1])).apply(values), values
    )


# ── cross-validation ─────────────────────────────────────────────────────
def _training_data(n_images: int = 6, pixels: int = 60) -> ClassTrainingData:
    rng = np.random.default_rng(6)
    features, labels, ranges = [], [], []
    offset = 0
    for _ in range(n_images):
        y = (rng.random(pixels) < 0.2).astype(np.uint8)
        x = np.stack([y + rng.normal(0, 0.3, pixels), rng.normal(0, 1, pixels)], axis=1)
        features.append(x.astype(np.float32))
        labels.append(y)
        ranges.append((offset, offset + pixels))
        offset += pixels
    return ClassTrainingData(
        cls="class_01",
        features=np.concatenate(features),
        labels=np.concatenate(labels),
        ranges=ranges,
        anomaly_types=[f"anomaly_{i % 3}" for i in range(n_images)],
        sample_ids=[f"s{i // 2}" for i in range(n_images)],
        feature_names=["signal", "noise"],
    )


@pytest.mark.parametrize("mode", ["loao", "sample"])
def test_every_pixel_gets_an_out_of_fold_prediction(mode):
    data = _training_data()
    predictions, labels = out_of_fold(
        data, "logreg", {"C": 1.0}, mode=mode, neg_per_pos=5
    )
    assert predictions.shape == labels.shape == data.labels.shape


def test_buckets_group_by_the_requested_key():
    data = _training_data()
    assert set(data.buckets("loao")) == {"anomaly_0", "anomaly_1", "anomaly_2"}
    assert set(data.buckets("sample")) == {"s0", "s1", "s2"}


def test_out_of_fold_beats_chance_on_a_learnable_signal():
    from spacepresso.core.metrics import average_precision

    data = _training_data(n_images=9, pixels=200)
    predictions, labels = out_of_fold(data, "logreg", {"C": 1.0}, neg_per_pos=10)
    assert average_precision(predictions, labels) > labels.mean() * 1.5


# ── fusion ───────────────────────────────────────────────────────────────
def test_mean_and_max_behave_as_named():
    scores = np.array([[0.2, 0.8]], dtype=np.float32)
    assert fusion.fuse(scores, "mean")[0] == pytest.approx(0.5)
    assert fusion.fuse(scores, "max")[0] == pytest.approx(0.8)


def test_geometric_fusion_punishes_disagreement():
    agree = np.array([[0.5, 0.5]], dtype=np.float32)
    disagree = np.array([[0.01, 0.99]], dtype=np.float32)
    assert fusion.fuse(disagree, "geometric")[0] < fusion.fuse(agree, "geometric")[0]


def test_weights_shift_the_result_towards_the_weighted_method():
    scores = np.array([[0.0, 1.0]], dtype=np.float32)
    assert fusion.fuse(scores, "mean", weights=[3, 1])[0] == pytest.approx(0.25)


def test_weights_are_rejected_where_they_are_meaningless():
    with pytest.raises(ValueError, match="does not accept weights"):
        fusion.fuse(np.zeros((1, 2), np.float32), "max", weights=[1, 1])


def test_wrong_weight_count_is_rejected():
    with pytest.raises(ValueError, match="3 weights for 2 methods"):
        fusion.fuse(np.zeros((1, 2), np.float32), "mean", weights=[1, 1, 1])


def test_rank_mean_neutralises_a_heavy_tailed_method():
    rng = np.random.default_rng(7)
    calm = rng.random((100, 1)).astype(np.float32)
    wild = (rng.random((100, 1)) * 1e6).astype(np.float32)
    scores = np.concatenate([calm, wild], axis=1)
    plain = fusion.fuse(scores, "mean")
    ranked = fusion.fuse(scores, "rank_mean")
    # The plain mean is essentially the wild method; the ranked one is not.
    assert abs(np.corrcoef(plain, wild[:, 0])[0, 1]) > 0.99
    assert abs(np.corrcoef(ranked, wild[:, 0])[0, 1]) < 0.95


def test_ecdf_transform_maps_to_percentiles():
    reference = np.arange(100, dtype=np.float32)
    assert fusion.ecdf_transform(np.array([50.0]), reference)[0] == pytest.approx(0.5)
