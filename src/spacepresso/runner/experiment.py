"""The experiment runner — written once, for every detector.

This module replaces 36 copies of ``main()`` (9,409 lines) and 14 of
``run_one_class()`` (1,419 lines), plus 16 ``write_submission``, 14
``RunConfig`` and 14 ``make_run_id``. Every one of those did the same nine
things in the same order, with small divergences that were bugs rather than
intent — one detector forgot to smooth before evaluating, another wrote its
ablation row before the submission so a crash in between left the two out of
sync.

The order, once:

1. scan the dataset and pick the classes to run
2. derive a content-hashed run id and create ``<report_dir>/runs/<run_id>/``
3. save ``config.json``, and mirror the log to ``run_log.txt``
4. per class — fit on ``train/good``, score ``train/anomaly``, report per-type
   pixel-AP, score ``test``
5. write ``local_eval.csv``
6. write ``local_predictions.npz`` and ``test_predictions.npz`` for the stacker
7. write ``submission.csv`` and ``submission.zip``
8. append a row to ``ablation_master.csv``
9. print the summary
"""

from __future__ import annotations

import csv
import math
import random
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from spacepresso.core.imaging import gaussian_smooth, resize_to_submission
from spacepresso.core.logging import (
    log_section,
    log_subsection,
    run_log,
    setup_logging,
)
from spacepresso.core.metrics import per_image_ap, pooled_pixel_ap
from spacepresso.core.predictions import PredictionWriter
from spacepresso.core.records import (
    TEST,
    TRAIN_ANOMALY,
    TRAIN_GOOD,
    ImageRecord,
    classes_in,
    scan_dataset,
    select,
)
from spacepresso.core.submission import ScoredRecord, write_submission
from spacepresso.core.tracking import append_to_master, make_run_id
from spacepresso.data.transforms import load_mask
from spacepresso.detectors.base import Detector, ScoreMaps
from spacepresso.runner.config import DetectorConfig, RuntimeConfig

__all__ = ["ClassResult", "RunResult", "run_experiment"]

logger = setup_logging()

#: Builds a fresh detector for one class. A factory rather than an instance
#: because several detectors hold per-class state (a memory bank, a trained
#: student) that must not leak between classes.
DetectorFactory = Callable[[], Detector]


@dataclass(slots=True)
class ClassResult:
    cls: str
    mean_ap: float
    pooled_ap: float
    eval_rows: list[dict[str, Any]] = field(default_factory=list)
    scored_test: list[ScoredRecord] = field(default_factory=list)
    elapsed_min: float = 0.0


@dataclass(slots=True)
class RunResult:
    run_id: str
    run_dir: Path
    class_aps: dict[str, float]
    overall_ap: float
    pooled_ap: float
    submission_path: Path | None
    elapsed_min: float


# ─────────────────────────────────────────────────────────────────────────────
# Per-class work
# ─────────────────────────────────────────────────────────────────────────────
def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _smooth_all(maps: ScoreMaps, sigma: float) -> ScoreMaps:
    return [gaussian_smooth(m, sigma) for m in maps]


def _evaluate(
    records: Sequence[ImageRecord],
    maps: ScoreMaps,
    writer: PredictionWriter | None,
) -> tuple[list[dict[str, Any]], float, float]:
    """Per-anomaly-type pixel-AP for one class, plus the pooled figure.

    Masks are loaded here, not by the detector — a detector should never see
    the labels it is being scored against.
    """
    by_type: dict[str, list[float]] = defaultdict(list)
    all_maps: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []

    for record, score in zip(records, maps, strict=True):
        mask = load_mask(record.mask_path, score.shape[0])
        if mask.shape != score.shape:
            mask = load_mask(record.mask_path, score.shape[0])
        by_type[record.anomaly_type or "?"].append(per_image_ap(score, mask))
        all_maps.append(score)
        all_masks.append(mask)
        if writer is not None:
            writer.add(record, score, mask)

    rows: list[dict[str, Any]] = []
    type_means: list[float] = []
    logger.info(
        "    %-16s %8s %26s", "anomaly_type", "n_views", "pixel-AP (mean ± std)"
    )
    for anomaly_type in sorted(by_type):
        values = np.asarray(by_type[anomaly_type], dtype=np.float64)
        type_means.append(float(values.mean()))
        logger.info(
            "    %-16s %8d %15.4f ± %.4f",
            anomaly_type,
            values.size,
            values.mean(),
            values.std(),
        )
        rows.append(
            {
                "anomaly_type": anomaly_type,
                "n_views": int(values.size),
                "ap_mean": float(values.mean()),
                "ap_std": float(values.std()),
                "ap_min": float(values.min()),
                "ap_max": float(values.max()),
            }
        )

    mean_ap = float(np.mean(type_means)) if type_means else float("nan")
    pooled = pooled_pixel_ap(all_maps, all_masks) if all_maps else float("nan")
    return rows, mean_ap, pooled


def _run_one_class(
    cls: str,
    records: Sequence[ImageRecord],
    make_detector: DetectorFactory,
    runtime: RuntimeConfig,
    config: DetectorConfig,
    writer: PredictionWriter | None,
) -> ClassResult:
    log_section(logger, f"CLASS {cls}", "─")
    started = time.time()

    train_good = select(records, cls=cls, split=TRAIN_GOOD)
    validation = select(records, cls=cls, split=TRAIN_ANOMALY)
    test = select(records, cls=cls, split=TEST)
    logger.info(
        "  train_good=%d  train_anomaly=%d  test=%d",
        len(train_good),
        len(validation),
        len(test),
    )

    if not train_good:
        logger.warning("%s has no train/good images — skipping", cls)
        return ClassResult(cls, float("nan"), float("nan"))

    detector = make_detector()
    try:
        detector.fit(train_good)

        eval_rows: list[dict[str, Any]] = []
        mean_ap = pooled_ap = float("nan")
        if not runtime.skip_eval and validation:
            log_subsection(logger, f"local validation  tta={config.tta}")
            maps = _smooth_all(detector.score(validation), config.smooth_sigma)
            eval_rows, mean_ap, pooled_ap = _evaluate(validation, maps, writer)
            for row in eval_rows:
                row["class"] = cls
            logger.info(
                "    >>> class %s  mean pixel-AP %.4f   pooled %.4f",
                cls,
                mean_ap,
                pooled_ap,
            )

        scored_test: list[ScoredRecord] = []
        if not runtime.skip_submission and test:
            log_subsection(logger, f"scoring {len(test)} test images")
            maps = _smooth_all(detector.score(test), config.smooth_sigma)
            scored_test = [
                (record, resize_to_submission(score))
                for record, score in zip(test, maps, strict=True)
            ]
    finally:
        detector.release()

    elapsed = (time.time() - started) / 60.0
    logger.info("  class %s done in %.1f min", cls, elapsed)
    return ClassResult(cls, mean_ap, pooled_ap, eval_rows, scored_test, elapsed)


# ─────────────────────────────────────────────────────────────────────────────
# Whole run
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(
    make_detector: Callable[[DetectorConfig, RuntimeConfig], Detector],
    config: DetectorConfig,
    runtime: RuntimeConfig,
) -> RunResult:
    """Run one detector over the dataset and write every artifact.

    Args:
        make_detector: ``(config, runtime) -> Detector``. Called once per
            class so per-class state cannot leak.
        config: the detector's own settings — hashed into the run id.
        runtime: paths and machine settings — deliberately not hashed.
    """
    _seed_everything(config.seed)
    started = time.time()

    records = scan_dataset(runtime.data_root)
    available = classes_in(records)
    classes = (
        [c for c in available if c in set(runtime.only_classes)]
        if runtime.only_classes
        else available
    )
    if not classes:
        raise ValueError(
            f"no classes to run. --only-classes={list(runtime.only_classes)} "
            f"matched none of {available}"
        )

    probe = make_detector(config, runtime)
    run_id = make_run_id(
        probe.slug(), probe.fingerprint(), run_tag=runtime.run_tag
    )
    detector_name = probe.name
    del probe

    run_dir = runtime.report_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with run_log(run_dir / "run_log.txt"):
        log_section(logger, f"{detector_name.upper()} — RUN {run_id}", "█")
        for key, value in config.to_dict().items():
            logger.info("  %-18s : %s", key, value)
        logger.info("  %-18s : %s", "device", runtime.torch_device)
        logger.info("  running %d class(es): %s", len(classes), ", ".join(classes))
        _write_config(run_dir, config, runtime, run_id, detector_name)

        writer = (
            PredictionWriter(with_masks=True)
            if runtime.save_predictions and not runtime.skip_eval
            else None
        )

        results: list[ClassResult] = []
        try:
            for cls in classes:
                results.append(
                    _run_one_class(
                        cls,
                        records,
                        lambda: make_detector(config, runtime),
                        runtime,
                        config,
                        writer,
                    )
                )
            if writer is not None and len(writer) > 0:
                writer.save(run_dir / "local_predictions.npz")
        finally:
            if writer is not None:
                writer.close()

        class_aps = {r.cls: r.mean_ap for r in results}
        overall = _log_summary(results)
        _write_eval_csv(run_dir, results)

        scored_test = [pair for r in results for pair in r.scored_test]
        submission_path: Path | None = None
        if scored_test:
            log_section(logger, "SUBMISSION", "=")
            if runtime.save_predictions:
                with PredictionWriter(with_masks=False) as test_writer:
                    for record, score in scored_test:
                        test_writer.add(record, score)
                    test_writer.save(run_dir / "test_predictions.npz")
            submission_path = write_submission(
                scored_test,
                run_dir / "submission.csv",
                zip_it=runtime.zip_submission,
            )
            logger.info("\n  Upload: %s", submission_path)

        elapsed = (time.time() - started) / 60.0
        _append_ablation_row(
            runtime.report_dir / "ablation_master.csv",
            run_id=run_id,
            detector=detector_name,
            config=config,
            runtime=runtime,
            class_aps=class_aps,
            overall_ap=overall,
            elapsed_min=elapsed,
            submission_path=submission_path,
        )
        log_section(logger, f"DONE — run_id={run_id}", "█")

    pooled = float(
        np.mean([r.pooled_ap for r in results if not math.isnan(r.pooled_ap)] or [np.nan])
    )
    return RunResult(
        run_id=run_id,
        run_dir=run_dir,
        class_aps=class_aps,
        overall_ap=overall,
        pooled_ap=pooled,
        submission_path=submission_path,
        elapsed_min=elapsed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Artifacts
# ─────────────────────────────────────────────────────────────────────────────
def _write_config(
    run_dir: Path,
    config: DetectorConfig,
    runtime: RuntimeConfig,
    run_id: str,
    detector: str,
) -> None:
    import json

    payload = {
        "run_id": run_id,
        "detector": detector,
        "config": config.to_dict(),
        "runtime": {
            "data_root": str(runtime.data_root),
            "report_dir": str(runtime.report_dir),
            "device": str(runtime.torch_device),
            "num_workers": runtime.num_workers,
            "batch_size": runtime.batch_size,
            "score_batch_size": runtime.score_batch_size,
            "amp": runtime.amp,
            "only_classes": list(runtime.only_classes),
            "run_tag": runtime.run_tag,
        },
        "fingerprint": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in config.fingerprint().items()
        },
    }
    (run_dir / "config.json").write_text(json.dumps(payload, indent=2, default=str))


def _log_summary(results: Sequence[ClassResult]) -> float:
    log_section(logger, "LOCAL VALIDATION SUMMARY", "=")
    logger.info("  %-12s %15s %12s", "class", "mean pixel-AP", "minutes")
    for result in results:
        logger.info(
            "  %-12s %15.4f %12.1f", result.cls, result.mean_ap, result.elapsed_min
        )
    valid = [r.mean_ap for r in results if not math.isnan(r.mean_ap)]
    overall = float(np.mean(valid)) if valid else float("nan")
    if valid:
        logger.info("  %-12s %15.4f", "OVERALL", overall)
    return overall


def _write_eval_csv(run_dir: Path, results: Sequence[ClassResult]) -> None:
    rows = [row for result in results for row in result.eval_rows]
    if not rows:
        return
    columns = ["class", "anomaly_type", "n_views", "ap_mean", "ap_std", "ap_min", "ap_max"]
    path = run_dir / "local_eval.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("  saved per-(class, anomaly_type) AP -> %s", path)


def _append_ablation_row(
    master_csv: Path,
    *,
    run_id: str,
    detector: str,
    config: DetectorConfig,
    runtime: RuntimeConfig,
    class_aps: dict[str, float],
    overall_ap: float,
    elapsed_min: float,
    submission_path: Path | None,
) -> None:
    """One row per run, with the same columns for every detector.

    Each detector previously built this dict by hand, so ``backbone`` meant
    different things in different rows and ``notes`` was free text. The
    detector-specific settings now go into one ``config`` column as a compact
    key=value string, which keeps the schema stable as detectors come and go.
    """
    settings = ",".join(
        f"{k}={v}" for k, v in sorted(config.to_dict().items()) if v not in (None, "")
    )
    row: dict[str, Any] = {
        "run_id": run_id,
        "run_tag": runtime.run_tag,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "detector": detector,
        "input_size": config.input_size,
        "tta": config.tta,
        "smooth_sigma": config.smooth_sigma,
        "seed": config.seed,
        "n_classes": len(class_aps),
        **{f"AP_{cls}": f"{ap:.4f}" for cls, ap in sorted(class_aps.items())},
        "AP_overall": f"{overall_ap:.4f}",
        "runtime_min": f"{elapsed_min:.1f}",
        "submission_path": str(submission_path or ""),
        "config": settings,
    }
    append_to_master(master_csv, row)
    logger.info("\n  ablation row appended -> %s", master_csv)
