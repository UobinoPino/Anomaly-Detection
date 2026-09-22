"""End-to-end: detectors run, artifacts are well-formed, the stacker fuses.

These use randomly-initialised backbones, because CI has no network access to
``download.pytorch.org``. They assert that the pipeline produces well-formed
output, not that the numbers are good — the numbers need real weights and a
real dataset.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spacepresso.core.codec import decode_to_float
from spacepresso.core.predictions import PredictionWriter, load_predictions
from spacepresso.core.submission import load_submission
from spacepresso.detectors import get_detector, list_detectors
from spacepresso.postprocess.tta import apply_tta

pytestmark = pytest.mark.torch


# ── the detector protocol ────────────────────────────────────────────────
def test_every_detector_declares_its_contract():
    for name in list_detectors():
        detector, config = get_detector(name)
        assert detector.name == name
        assert detector.config_type is config
        assert hasattr(detector, "fit") and hasattr(detector, "score")


def test_every_config_produces_a_fingerprint_and_a_slug():
    for name in list_detectors():
        _detector, config_class = get_detector(name)
        config = config_class()
        assert config.fingerprint()
        assert all(isinstance(part, str) for part in config.slug_parts())


def test_config_rejects_an_invalid_tta_mode():
    _detector, config_class = get_detector("patchcore")
    with pytest.raises(ValueError, match="tta must be one of"):
        config_class(tta="rotate90")


def test_input_size_is_validated_against_the_patch_stride():
    _detector, config_class = get_detector("patchcore")
    with pytest.raises(ValueError, match="multiple of 14"):
        config_class(backbone="dinov2_vits14", input_size=100)


# ── TTA ──────────────────────────────────────────────────────────────────
def test_tta_none_calls_the_scorer_once():
    calls = []

    def score(x):
        calls.append(x)
        return x.mean(dim=1)

    apply_tta(score, torch.randn(2, 3, 8, 8), "none")
    assert len(calls) == 1


@pytest.mark.parametrize("mode, expected", [("hflip", 2), ("hvflip", 3), ("d4", 8)])
def test_tta_augmentation_counts(mode, expected):
    calls = []

    def score(x):
        calls.append(x)
        return x.mean(dim=1)

    apply_tta(score, torch.randn(1, 3, 8, 8), mode)
    assert len(calls) == expected


def test_tta_inverts_its_augmentations():
    """A scorer that returns a spatially distinctive map must come back in the
    original orientation, or the averaging blurs across flips."""
    marker = torch.zeros(1, 3, 8, 8)
    marker[0, :, 0, 0] = 1.0

    def score(x):
        return x[:, 0]

    for mode in ("hflip", "vflip", "hvflip", "d4"):
        out = apply_tta(score, marker, mode)
        assert out[0].argmax().item() == 0, f"{mode} did not invert"


def test_unknown_tta_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown TTA mode"):
        apply_tta(lambda x: x[:, 0], torch.zeros(1, 3, 4, 4), "flip-diagonal")


# ── prediction writer ────────────────────────────────────────────────────
def test_prediction_writer_round_trips(tmp_path, dataset):
    from spacepresso.core.records import TRAIN_ANOMALY, scan_dataset, select

    records = select(scan_dataset(dataset), split=TRAIN_ANOMALY)[:4]
    rng = np.random.default_rng(0)

    with PredictionWriter(with_masks=True) as writer:
        for record in records:
            writer.add(record, rng.random((16, 16)), rng.integers(0, 2, (16, 16)))
        path = writer.save(tmp_path / "local_predictions.npz")

    loaded = load_predictions(path)
    assert loaded["scores"].shape == (4, 16, 16)
    assert loaded["masks"].shape == (4, 16, 16)
    assert set(np.unique(loaded["masks"]).tolist()) <= {0, 1}
    assert list(loaded["ids"]) == [r.stem for r in records]


def test_prediction_writer_stores_scores_unmodified(tmp_path, dataset):
    """Detectors emit unbounded scores and the upper tail *is* the signal.
    Normalising here would destroy what the stacker is given raw maps for."""
    from spacepresso.core.records import TRAIN_ANOMALY, scan_dataset, select

    record = select(scan_dataset(dataset), split=TRAIN_ANOMALY)[0]
    scores = np.array([[-5.0, 1000.0], [0.0, 3.0]], dtype=np.float32)

    with PredictionWriter(with_masks=False) as writer:
        writer.add(record, scores)
        path = writer.save(tmp_path / "test_predictions.npz")

    assert np.array_equal(load_predictions(path)["scores"][0], scores)


def test_prediction_writer_replaces_non_finite_values(tmp_path, dataset):
    from spacepresso.core.records import TRAIN_ANOMALY, scan_dataset, select

    record = select(scan_dataset(dataset), split=TRAIN_ANOMALY)[0]
    scores = np.array([[np.nan, 1.0], [np.inf, 0.0]], dtype=np.float32)

    with PredictionWriter(with_masks=False) as writer:
        writer.add(record, scores)
        path = writer.save(tmp_path / "p.npz")

    assert np.isfinite(load_predictions(path)["scores"]).all()


def test_saving_nothing_raises():
    with PredictionWriter(with_masks=False) as writer, pytest.raises(RuntimeError):
        writer.save(__import__("pathlib").Path("unused.npz"))


# ── full runs ────────────────────────────────────────────────────────────
DETECTOR_SETTINGS = {
    "patchcore": dict(backbone="resnet18", feature_layers=(2, 3), coreset_frac=0.3),
    "efficientad": dict(
        teacher_backbone="resnet18",
        total_iters=2,
        train_batch_size=2,
        norm_stat_images=4,
    ),
    "fastflow": dict(
        backbone="resnet18",
        feature_layers=(2, 3),
        total_iters=2,
        train_batch_size=2,
        n_blocks=2,
        norm_stat_images=4,
    ),
    "cfa": dict(
        backbone="resnet18",
        feature_layers=(2, 3),
        total_iters=2,
        train_batch_size=2,
        coreset_frac=0.2,
        memory_refresh_every=2,
    ),
    "draem": dict(base_channels=8, total_iters=2, train_batch_size=2),
    "cutpaste": dict(
        backbone="resnet18",
        feature_layers=(2, 3),
        total_iters=2,
        train_batch_size=2,
        coreset_frac=0.3,
    ),
    "glass": dict(
        backbone="resnet18",
        feature_layers=(2, 3),
        total_iters=2,
        train_batch_size=2,
        warmup_iters=1,
        discriminator_hidden=32,
    ),
    "uniad": dict(
        backbone="resnet18",
        feature_layer=3,
        total_iters=2,
        train_batch_size=2,
        model_dim=32,
        n_heads=4,
        neighbour_radius=0,
        n_encoder_layers=1,
        n_decoder_layers=1,
        norm_stat_images=4,
    ),
    "reverse_distillation": dict(
        teacher_backbone="wide_resnet50_2",
        teacher_layers=(1, 2, 3),
        total_iters=2,
        train_batch_size=2,
    ),
    "textad": dict(base_channels=8, depth=3, total_iters=2, train_batch_size=2),
    "transfusion": dict(
        base_channels=8,
        time_dim=16,
        total_iters=2,
        train_batch_size=2,
        texture_pool_size=4,
    ),
}


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(DETECTOR_SETTINGS))
def test_detector_runs_end_to_end(name, tmp_path, dataset):
    from fixtures.harness import run_detector

    result = run_detector(
        name, root=tmp_path, data_root=dataset, **DETECTOR_SETTINGS[name]
    )

    produced = {p.name for p in result.run_dir.iterdir()}
    assert {
        "config.json",
        "run_log.txt",
        "local_eval.csv",
        "local_predictions.npz",
        "test_predictions.npz",
        "submission.csv",
        "submission.zip",
    } <= produced

    submission = load_submission(result.run_dir / "submission.csv")
    assert len(submission) == 12  # 2 classes x 3 samples x 2 views
    matrix = decode_to_float(next(iter(submission.values())))
    assert matrix.shape == (224, 224)
    assert matrix.min() >= 0.0 and matrix.max() <= 1.0
    assert np.isfinite(result.overall_ap)


@pytest.mark.slow
def test_run_id_tracks_the_science_not_the_label(tmp_path, dataset):
    from fixtures.harness import run_detector

    base = dict(backbone="resnet18", feature_layers=(2, 3), coreset_frac=0.3)
    a = run_detector("patchcore", root=tmp_path / "a", data_root=dataset, **base)
    b = run_detector("patchcore", root=tmp_path / "b", data_root=dataset, **base)
    c = run_detector(
        "patchcore",
        root=tmp_path / "c",
        data_root=dataset,
        **{**base, "coreset_frac": 0.5},
    )

    assert a.run_id.split("_")[-1] == b.run_id.split("_")[-1]
    assert a.run_id.split("_")[-1] != c.run_id.split("_")[-1]


@pytest.mark.slow
def test_stacker_fuses_three_detectors(tmp_path, dataset):
    from fixtures.harness import run_detector
    from spacepresso.stacking import FeatureConfig, StackerConfig, run_stacker

    runs = [
        run_detector(
            name, root=tmp_path, data_root=dataset, **DETECTOR_SETTINGS[name]
        ).run_dir
        for name in ("patchcore", "efficientad", "draem")
    ]

    result = run_stacker(
        StackerConfig(
            runs=tuple(runs),
            data_root=dataset,
            report_dir=tmp_path / "stack",
            estimator="xgboost",
            features=FeatureConfig(gaussian_sigmas=(2.0,), window_sizes=(3,)),
        )
    )

    assert np.isfinite(result.overall_ap)
    assert result.submission_path is not None
    assert len(load_submission(result.run_dir / "submission.csv")) == 12


def test_stacking_a_single_run_is_refused(tmp_path, dataset):
    from spacepresso.stacking import StackerConfig

    with pytest.raises(ValueError, match="two or more runs"):
        StackerConfig(runs=(tmp_path,), report_dir=tmp_path)
