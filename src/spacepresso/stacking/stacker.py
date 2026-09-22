"""The stacker: fit per-class models over several detectors, then fuse.

Replaces ``xgboost_stacker_v3`` … ``v8``, ``log_stacker_v2``,
``logreg_stacker`` and ``ensemble_tier0`` — about 17,500 lines of forks of one
another. Each new version was a full copy of the previous file with a few
hundred lines changed, and ``v8`` additionally imported roughly sixty symbols
back out of ``v6`` and shadowed several of them.

What actually varied between those versions is now configuration: which
estimator, which features, which rank-normalisation scope, which
cross-validation grouping, whether to tune.

Pipeline:

1. load and align each detector's validation predictions
2. rank-normalise, so incomparable score scales become comparable
3. build per-pixel features
4. per class: cross-validate, optionally tune, fit a final model
5. calibrate the outputs on the out-of-fold predictions
6. apply to the test set and write one fused submission
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from spacepresso.core.logging import log_section, log_subsection, run_log, setup_logging
from spacepresso.core.metrics import average_precision
from spacepresso.core.records import ImageRecord, parse_view, scan_dataset
from spacepresso.core.submission import write_submission
from spacepresso.core.tracking import append_to_master, make_run_id
from spacepresso.postprocess.spatial import (
    load_spatial_priors,
    suppress_small_components,
)
from spacepresso.stacking.calibration import CalibrationMethod, fit_calibrator
from spacepresso.stacking.crossval import (
    ClassTrainingData,
    CVMode,
    out_of_fold,
    pooled_cv_score,
)
from spacepresso.stacking.dataset import (
    AlignedValidation,
    load_test_scores,
    load_validation,
)
from spacepresso.stacking.estimators import (
    ESTIMATORS,
    EstimatorName,
    build_estimator,
    default_params,
    suggest_params,
)
from spacepresso.stacking.features import FeatureConfig, featurize_image
from spacepresso.stacking.normalisation import Scope, rank_normalise

__all__ = ["StackerConfig", "StackerResult", "run_stacker"]

logger = setup_logging()


@dataclass
class StackerConfig:
    """Everything the stacker needs. One config, not six programs."""

    runs: tuple[Path, ...] = ()
    data_root: Path | None = None
    report_dir: Path = Path("baseline_out")
    prior_dir: Path | None = None

    estimator: EstimatorName = "xgboost"
    features: FeatureConfig = field(default_factory=FeatureConfig)

    val_scope: Scope = "per_class"
    test_scope: Scope = "per_class"
    cv_mode: CVMode = "loao"
    neg_per_pos: int = 30
    calibrate: CalibrationMethod = "isotonic"

    tune_trials: int = 0
    tune_timeout: int | None = None
    per_class_params: bool = False

    min_component_size: int = 0
    device: str = "cpu"
    seed: int = 0
    run_tag: str = ""
    only_classes: tuple[str, ...] = ()
    zip_submission: bool = True

    def __post_init__(self) -> None:
        if self.estimator not in ESTIMATORS:
            raise ValueError(
                f"estimator must be one of {ESTIMATORS}, got {self.estimator!r}"
            )
        self.runs = tuple(Path(r) for r in self.runs)
        if not self.runs:
            raise ValueError("the stacker needs at least two run directories")
        if len(self.runs) < 2:
            raise ValueError(
                "stacking one method is just that method — pass two or more runs"
            )

    def fingerprint(self) -> dict[str, Any]:
        return {
            "estimator": self.estimator,
            "features": asdict(self.features),
            "val_scope": self.val_scope,
            "test_scope": self.test_scope,
            "cv_mode": self.cv_mode,
            "neg_per_pos": self.neg_per_pos,
            "calibrate": self.calibrate,
            "tune_trials": self.tune_trials,
            "per_class_params": self.per_class_params,
            "min_component_size": self.min_component_size,
            "seed": self.seed,
            "methods": [r.name for r in self.runs],
        }

    def slug(self) -> str:
        parts = [
            "stack",
            self.estimator,
            f"m{len(self.runs)}",
            f"rn-{self.val_scope}",
            self.cv_mode,
        ]
        if self.calibrate != "none":
            parts.append(f"cal-{self.calibrate}")
        if self.tune_trials:
            parts.append(f"tune{self.tune_trials}")
        return "_".join(parts)


@dataclass(slots=True)
class StackerResult:
    run_id: str
    run_dir: Path
    class_ap: dict[str, float]
    overall_ap: float
    submission_path: Path | None
    elapsed_min: float


# ─────────────────────────────────────────────────────────────────────────────
# Feature building
# ─────────────────────────────────────────────────────────────────────────────
def _build_class_data(
    validation: AlignedValidation,
    cls: str,
    config: StackerConfig,
    priors: dict[str, npt.NDArray[np.float32]],
    all_classes: Sequence[str],
) -> ClassTrainingData | None:
    indices = validation.indices_for(cls)
    if indices.size == 0:
        return None

    cache: dict = {}
    blocks: list[npt.NDArray[np.float32]] = []
    labels: list[npt.NDArray[np.uint8]] = []
    ranges: list[tuple[int, int]] = []
    names: list[str] = []
    offset = 0

    for index in indices:
        scores = [
            validation.scores[index, :, :, method]
            for method in range(validation.n_methods)
        ]
        matrix, names = featurize_image(
            scores,
            config.features,
            class_id=cls,
            all_classes=all_classes,
            view=int(validation.views[index]),
            prior=priors.get(cls),
            cache=cache,
        )
        blocks.append(matrix)
        labels.append((validation.masks[index] > 0).astype(np.uint8).ravel())
        ranges.append((offset, offset + matrix.shape[0]))
        offset += matrix.shape[0]

    data = ClassTrainingData(
        cls=cls,
        features=np.concatenate(blocks, axis=0),
        labels=np.concatenate(labels, axis=0),
        ranges=ranges,
        anomaly_types=[validation.anomaly_types[i] for i in indices],
        sample_ids=[validation.sample_ids[i] for i in indices],
        feature_names=names,
    )
    logger.info(
        "    %s: %d images, %d pixels, %d features, %.2f%% positive",
        cls,
        data.n_images,
        data.labels.size,
        data.features.shape[1],
        100.0 * data.n_positive / max(data.labels.size, 1),
    )
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Tuning
# ─────────────────────────────────────────────────────────────────────────────
def _tune(
    data_by_class: dict[str, ClassTrainingData], config: StackerConfig
) -> dict[str, Any]:
    """Optuna search against the pooled out-of-fold AP.

    Tuning against the same pooled metric the leaderboard uses, under the same
    grouped cross-validation, is the whole point — the earlier stackers tuned
    against per-image AP and then wondered why the leaderboard disagreed.
    """
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "tuning needs optuna. Install it with: pip install optuna"
        ) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: Any) -> float:
        params = suggest_params(config.estimator, trial)
        if config.estimator == "xgboost":
            params.update(
                {
                    k: v
                    for k, v in default_params("xgboost", device=config.device).items()
                    if k in ("tree_method", "eval_metric", "n_jobs", "device")
                }
            )
        scores = [
            pooled_cv_score(
                data,
                config.estimator,
                params,
                seed=config.seed,
                mode=config.cv_mode,
                neg_per_pos=config.neg_per_pos,
            )
            for data in data_by_class.values()
        ]
        finite = [s for s in scores if np.isfinite(s)]
        return float(np.mean(finite)) if finite else 0.0

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.seed)
    )
    study.optimize(objective, n_trials=config.tune_trials, timeout=config.tune_timeout)

    best = dict(study.best_params)
    if config.estimator == "xgboost":
        defaults = default_params("xgboost", device=config.device)
        for key in ("tree_method", "eval_metric", "n_jobs", "device"):
            best.setdefault(key, defaults[key])
    logger.info("    best pooled OOF AP %.4f with %s", study.best_value, best)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
def run_stacker(config: StackerConfig) -> StackerResult:
    """Fit the stacker and write a fused submission."""
    started = time.time()
    run_id = make_run_id(config.slug(), config.fingerprint(), run_tag=config.run_tag)
    run_dir = config.report_dir / "stacks" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with run_log(run_dir / "run_log.txt"):
        log_section(logger, f"STACKER — RUN {run_id}", "█")
        logger.info("  estimator   : %s", config.estimator)
        logger.info("  methods     : %d", len(config.runs))
        for run in config.runs:
            logger.info("      %s", run.name)
        logger.info("  val scope   : %s", config.val_scope)
        logger.info("  cv mode     : %s", config.cv_mode)
        logger.info("  calibration : %s", config.calibrate)
        (run_dir / "config.json").write_text(
            json.dumps(config.fingerprint(), indent=2, default=str)
        )

        log_section(logger, "LOADING", "=")
        validation = load_validation(
            [run / "local_predictions.npz" for run in config.runs],
            [run.name for run in config.runs],
        )
        classes = validation.classes_present()
        if config.only_classes:
            classes = [c for c in classes if c in set(config.only_classes)]

        logger.info("  rank-normalising validation scores (scope=%s)", config.val_scope)
        rank_normalise(
            validation.scores,
            config.val_scope,
            classes=validation.classes.tolist(),
            views=validation.views.tolist(),
            inplace=True,
        )

        priors = load_spatial_priors(config.prior_dir, classes)

        log_section(logger, "FEATURES", "=")
        data_by_class: dict[str, ClassTrainingData] = {}
        for cls in classes:
            data = _build_class_data(validation, cls, config, priors, classes)
            if data is None or data.n_positive == 0:
                logger.warning("  %s has no positive pixels — skipping", cls)
                continue
            data_by_class[cls] = data

        if not data_by_class:
            raise RuntimeError("no class had usable validation data")

        params = default_params(config.estimator, device=config.device)
        if config.tune_trials:
            log_section(logger, f"TUNING ({config.tune_trials} trials)", "=")
            params = _tune(data_by_class, config)

        log_section(logger, "FITTING", "=")
        models: dict[str, Any] = {}
        calibrators: dict[str, Any] = {}
        class_ap: dict[str, float] = {}

        for cls, data in data_by_class.items():
            log_subsection(logger, f"class {cls}")
            predictions, labels = out_of_fold(
                data,
                config.estimator,
                params,
                seed=config.seed,
                mode=config.cv_mode,
                neg_per_pos=config.neg_per_pos,
            )
            class_ap[cls] = average_precision(predictions, labels)
            logger.info("    out-of-fold pooled pixel-AP: %.4f", class_ap[cls])

            calibrators[cls] = fit_calibrator(
                config.calibrate, predictions, labels, seed=config.seed
            )
            models[cls] = build_estimator(config.estimator, params, config.seed).fit(
                data.features, data.labels
            )
            _log_importance(models[cls], data.feature_names)
            # The feature matrix for one class is several GB; drop it now
            # rather than holding eight of them until the loop ends.
            data.features = np.empty((0, 0), dtype=np.float32)

        overall = float(np.mean(list(class_ap.values()))) if class_ap else float("nan")
        _log_summary(class_ap, overall)

        submission_path = _fuse_test(
            config, run_dir, models, calibrators, priors, classes, validation
        )

        elapsed = (time.time() - started) / 60.0
        append_to_master(
            config.report_dir / "stack_master.csv",
            {
                "run_id": run_id,
                "run_tag": config.run_tag,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "estimator": config.estimator,
                "n_methods": len(config.runs),
                "methods": "|".join(r.name for r in config.runs),
                "val_scope": config.val_scope,
                "cv_mode": config.cv_mode,
                "calibrate": config.calibrate,
                **{f"AP_{cls}": f"{ap:.4f}" for cls, ap in sorted(class_ap.items())},
                "AP_overall": f"{overall:.4f}",
                "runtime_min": f"{elapsed:.1f}",
                "submission_path": str(submission_path or ""),
            },
        )
        log_section(logger, f"DONE — run_id={run_id}", "█")

    return StackerResult(
        run_id=run_id,
        run_dir=run_dir,
        class_ap=class_ap,
        overall_ap=overall,
        submission_path=submission_path,
        elapsed_min=elapsed,
    )


def _log_importance(estimator: Any, names: Sequence[str], top: int = 12) -> None:
    importance = estimator.feature_importance()
    if importance is None or len(importance) != len(names):
        return
    order = np.argsort(importance)[::-1][:top]
    logger.info("    top features: %s", ", ".join(f"{names[i]}" for i in order))


def _log_summary(class_ap: dict[str, float], overall: float) -> None:
    log_section(logger, "OUT-OF-FOLD SUMMARY", "=")
    logger.info("  %-14s %12s", "class", "pooled AP")
    for cls, ap in sorted(class_ap.items()):
        logger.info("  %-14s %12.4f", cls, ap)
    logger.info("  %-14s %12.4f", "OVERALL", overall)


def _fuse_test(
    config: StackerConfig,
    run_dir: Path,
    models: dict[str, Any],
    calibrators: dict[str, Any],
    priors: dict[str, npt.NDArray[np.float32]],
    classes: Sequence[str],
    validation: AlignedValidation,
) -> Path | None:
    """Apply the per-class models to the test submissions and write the fusion."""
    submissions = [run / "submission.csv" for run in config.runs]
    missing = [p for p in submissions if not p.exists()]
    if missing:
        logger.warning(
            "  skipping test fusion: %d run(s) have no submission.csv (%s)",
            len(missing),
            ", ".join(p.parent.name for p in missing),
        )
        return None

    log_section(logger, "TEST FUSION", "=")
    ids, scores_u8 = load_test_scores(submissions)
    class_map = _class_map(config, ids)

    scores = scores_u8.astype(np.float32) / 255.0
    del scores_u8
    test_classes = [class_map.get(image_id, classes[0]) for image_id in ids]
    test_views = [
        (lambda v: v if v is not None else -1)(_view_of(image_id)) for image_id in ids
    ]
    rank_normalise(
        scores,
        config.test_scope,
        classes=test_classes,
        views=test_views,
        inplace=True,
    )

    cache: dict = {}
    scored: list[tuple[ImageRecord, npt.NDArray[np.float32]]] = []
    height, width = scores.shape[1:3]

    for position, image_id in enumerate(ids):
        cls = test_classes[position]
        model = models.get(cls)
        if model is None:
            # A class with no validation positives has no model. Falling back
            # to the cross-method mean keeps its rows present and ranked
            # rather than dropping them from the submission.
            fused = scores[position].mean(axis=-1)
        else:
            matrix, _ = featurize_image(
                [scores[position, :, :, m] for m in range(scores.shape[-1])],
                config.features,
                class_id=cls,
                all_classes=classes,
                view=test_views[position],
                prior=priors.get(cls),
                cache=cache,
            )
            fused = calibrators[cls].apply(model.predict(matrix)).reshape(height, width)

        if config.min_component_size > 0:
            fused = suppress_small_components(fused, config.min_component_size)

        scored.append((_pseudo_record(image_id, cls), fused.astype(np.float32)))

        if (position + 1) % 500 == 0:
            logger.info("    fused %d/%d", position + 1, len(ids))

    # Stacker outputs are calibrated probabilities already in [0, 1], so the
    # global percentile rescale that detectors need would only compress them.
    return write_submission(
        scored,
        run_dir / "submission.csv",
        calibrate=False,
        zip_it=config.zip_submission,
    )


def _view_of(image_id: str) -> int | None:
    _, view = parse_view(f"{image_id}.png")
    return view


def _class_map(config: StackerConfig, ids: Sequence[str]) -> dict[str, str]:
    """Map test image IDs to their class, from the dataset directory layout."""
    if config.data_root is None:
        return {}
    records = scan_dataset(config.data_root)
    return {record.stem: record.cls for record in records}


def _pseudo_record(image_id: str, cls: str) -> ImageRecord:
    """A record carrying just the identity the submission writer needs."""
    return ImageRecord(path=Path(f"{image_id}.png"), cls=cls, split="test")
