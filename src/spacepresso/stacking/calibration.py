"""Probability calibration for stacker outputs.

A gradient-boosted model trained on pixels where positives are ~1% produces
scores that rank well but are not calibrated probabilities. That does not
matter for AP — which is rank-based — but it matters a great deal when two
models' outputs are averaged, and it matters for reading the numbers at all.

Two calibrators:

* **isotonic** — a monotone step function fitted to the out-of-fold
  predictions. Flexible, and monotone, so it cannot change the ranking within
  a model.
* **platt** — a logistic fit. Two parameters, so far less prone to
  overfitting on small out-of-fold sets.

Large fits are downsampled with positive-preserving stratification.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from spacepresso.core.logging import get_logger

__all__ = ["METHODS", "CalibrationMethod", "Calibrator", "fit_calibrator"]

logger = get_logger(__name__)

CalibrationMethod = Literal["none", "isotonic", "platt"]
METHODS: tuple[CalibrationMethod, ...] = ("none", "isotonic", "platt")


class Calibrator:
    """A fitted mapping from raw model output to a calibrated probability."""

    def __init__(self, method: CalibrationMethod, model: Any = None) -> None:
        self.method = method
        self.model = model

    def __call__(
        self, predictions: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.float32]:
        return self.apply(predictions)

    def apply(self, predictions: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
        values = np.asarray(predictions, dtype=np.float64).ravel()
        if self.method == "none" or self.model is None:
            return values.astype(np.float32)
        if self.method == "isotonic":
            return self.model.predict(values).astype(np.float32)
        # platt: a logistic regression on the single raw score
        return self.model.predict_proba(values.reshape(-1, 1))[:, 1].astype(np.float32)


def _stratified_sample(
    predictions: npt.NDArray[np.floating],
    labels: npt.NDArray[np.integer],
    max_rows: int,
    seed: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int8]]:
    """Downsample, keeping every positive that fits.

    Positives are the scarce, informative class: the calibration curve's shape
    in the high-score region is entirely determined by them. Negatives are
    interchangeable, so they take whatever budget remains.
    """
    positives = np.flatnonzero(labels > 0)
    negatives = np.flatnonzero(labels == 0)
    if predictions.size <= max_rows:
        return predictions.astype(np.float64), (labels > 0).astype(np.int8)

    rng = np.random.default_rng(seed)
    # Target a 1:9 split, but never discard positives to hit it.
    n_positive = min(positives.size, max(max_rows // 10, 1))
    n_negative = min(negatives.size, max_rows - n_positive)

    keep = np.concatenate(
        [
            rng.choice(positives, size=n_positive, replace=False),
            rng.choice(negatives, size=n_negative, replace=False),
        ]
    )
    return predictions[keep].astype(np.float64), (labels[keep] > 0).astype(np.int8)


def fit_calibrator(
    method: CalibrationMethod,
    predictions: npt.NDArray[np.floating],
    labels: npt.NDArray[np.integer],
    *,
    max_rows: int = 2_000_000,
    seed: int = 0,
) -> Calibrator:
    """Fit a calibrator on out-of-fold predictions.

    Args:
        method: ``none``, ``isotonic`` or ``platt``.
        predictions: raw out-of-fold model outputs.
        labels: matching binary labels.
        max_rows: subsample cap, applied with positive-preserving stratification.
    """
    if method not in METHODS:
        raise ValueError(f"calibration method must be one of {METHODS}, got {method!r}")
    if method == "none":
        return Calibrator("none")

    flat_predictions = np.asarray(predictions, dtype=np.float64).ravel()
    flat_labels = np.asarray(labels).ravel()
    if flat_predictions.shape != flat_labels.shape:
        raise ValueError(
            f"prediction/label shape mismatch: "
            f"{flat_predictions.shape} vs {flat_labels.shape}"
        )

    n_positive = int((flat_labels > 0).sum())
    if n_positive == 0 or n_positive == flat_labels.size:
        logger.warning(
            "calibration skipped: out-of-fold labels are single-class (%d positives "
            "of %d)",
            n_positive,
            flat_labels.size,
        )
        return Calibrator("none")

    sampled_predictions, sampled_labels = _stratified_sample(
        flat_predictions, flat_labels, max_rows, seed
    )
    logger.info(
        "    calibrating (%s) on %d rows (%d positive, %.2f%%)",
        method,
        sampled_labels.size,
        int(sampled_labels.sum()),
        100.0 * sampled_labels.mean(),
    )

    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(sampled_predictions, sampled_labels)
    else:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000)
        model.fit(sampled_predictions.reshape(-1, 1), sampled_labels)

    return Calibrator(method, model)
