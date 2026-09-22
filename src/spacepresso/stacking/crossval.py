"""Cross-validation for the stacker.

The validation set is small — a few hundred anomalous images per class — and
its images are not independent: the same physical object appears in several
views, and images of the same anomaly type share a defect mechanism. A plain
random split leaks across both, and reports an AP a few points above what the
leaderboard gives.

Two grouping schemes address that:

``loao`` (leave-one-anomaly-type-out)
    Hold out every image of one anomaly type at a time. Answers "does this
    generalise to a defect mechanism it has not seen?", which is the harder
    and more honest question.
``sample``
    Hold out all views of one physical object at a time. Answers "does this
    generalise to a new object?" — easier, but the right choice when the
    anomaly types are few and holding one out removes too much data.

Negative subsampling
--------------------
Pixels are ~99% negative. Training on all of them is slow and adds nothing
past a point, so training folds keep every positive and a bounded multiple of
negatives. Prediction always covers every pixel of the held-out images — the
out-of-fold scores feed the calibrator and the reported AP, and subsampling
those would bias both.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

from spacepresso.core.metrics import average_precision
from spacepresso.stacking.estimators import Estimator, build_estimator

__all__ = ["CV_MODES", "CVMode", "ClassTrainingData", "out_of_fold", "pooled_cv_score"]

CVMode = Literal["loao", "sample"]
CV_MODES: tuple[CVMode, ...] = ("loao", "sample")


class ClassTrainingData:
    """One class's pixel features, labels and the image boundaries within them.

    Features for a class are one tall matrix rather than a list of per-image
    matrices: at 224x224 over a few hundred images that is tens of millions of
    rows, and a Python-level list of arrays costs both memory and time. The
    ``ranges`` index recovers image boundaries when a fold needs them.
    """

    def __init__(
        self,
        cls: str,
        features: npt.NDArray[np.float32],
        labels: npt.NDArray[np.uint8],
        ranges: Sequence[tuple[int, int]],
        anomaly_types: Sequence[str],
        sample_ids: Sequence[str],
        feature_names: Sequence[str],
    ) -> None:
        self.cls = cls
        self.features = features
        self.labels = labels
        self.ranges = list(ranges)
        self.anomaly_types = list(anomaly_types)
        self.sample_ids = list(sample_ids)
        self.feature_names = list(feature_names)

    @property
    def n_images(self) -> int:
        return len(self.ranges)

    @property
    def n_positive(self) -> int:
        return int((self.labels > 0).sum())

    def buckets(self, mode: CVMode) -> dict[str, list[int]]:
        """Image indices grouped into held-out folds."""
        keys = self.anomaly_types if mode == "loao" else self.sample_ids
        groups: dict[str, list[int]] = {}
        for index, key in enumerate(keys):
            groups.setdefault(str(key), []).append(index)
        return groups


def _training_rows(
    data: ClassTrainingData,
    image_indices: Sequence[int],
    neg_per_pos: int,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]] | None:
    """Gather subsampled training rows from the given images.

    Subsampling is per image, not pooled: pooling would let a few
    positive-dense images dominate the negative sample and leave the rest of
    the class's background unrepresented.
    """
    feature_blocks: list[npt.NDArray[np.float32]] = []
    label_blocks: list[npt.NDArray[np.uint8]] = []

    for index in image_indices:
        start, end = data.ranges[index]
        labels = data.labels[start:end]
        positives = np.flatnonzero(labels > 0)
        if positives.size == 0:
            continue
        negatives = np.flatnonzero(labels == 0)
        n_keep = min(negatives.size, positives.size * neg_per_pos)
        sampled = (
            rng.choice(negatives, n_keep, replace=False)
            if n_keep < negatives.size
            else negatives
        )
        keep = np.concatenate([positives, sampled])
        feature_blocks.append(data.features[start:end][keep])
        label_blocks.append(labels[keep])

    if not feature_blocks:
        return None
    return np.concatenate(feature_blocks), np.concatenate(label_blocks)


def out_of_fold(
    data: ClassTrainingData,
    estimator_name: str,
    params: dict,
    *,
    seed: int = 0,
    mode: CVMode = "loao",
    neg_per_pos: int = 30,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
    """Out-of-fold predictions for every pixel of every image in the class.

    Returns ``(predictions, labels)`` covering only the pixels that ended up
    in some held-out fold — which is all of them unless a fold had no
    trainable data.
    """
    rng = np.random.default_rng(seed)
    predictions = np.full(data.labels.shape, np.nan, dtype=np.float32)

    for held_out in data.buckets(mode).values():
        held = set(held_out)
        training_images = [i for i in range(data.n_images) if i not in held]
        rows = _training_rows(data, training_images, neg_per_pos, rng)
        if rows is None:
            continue

        estimator: Estimator = build_estimator(estimator_name, params, seed).fit(*rows)
        for index in held_out:
            start, end = data.ranges[index]
            predictions[start:end] = estimator.predict(data.features[start:end])

    scored = ~np.isnan(predictions)
    return predictions[scored], data.labels[scored].astype(np.uint8)


def pooled_cv_score(
    data: ClassTrainingData,
    estimator_name: str,
    params: dict,
    *,
    seed: int = 0,
    mode: CVMode = "loao",
    neg_per_pos: int = 30,
) -> float:
    """Pooled pixel-AP of the out-of-fold predictions for one class.

    Pooled, not per-image averaged, because that is what the leaderboard
    measures — tuning against the per-image average selects models whose
    scores are not comparable between images.
    """
    predictions, labels = out_of_fold(
        data,
        estimator_name,
        params,
        seed=seed,
        mode=mode,
        neg_per_pos=neg_per_pos,
    )
    if predictions.size == 0:
        return float("nan")
    return average_precision(predictions, labels)
