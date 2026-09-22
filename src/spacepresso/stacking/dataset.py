"""Loading and aligning the detectors' predictions.

Each detector run writes ``local_predictions.npz`` (validation scores plus
ground-truth masks) and ``submission.csv`` (test scores, q8rle-encoded). The
stacker needs those aligned: the same images, in the same order, at the same
resolution, across every method.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from spacepresso.core import codec
from spacepresso.core.imaging import resize_nearest
from spacepresso.core.logging import get_logger
from spacepresso.core.predictions import load_predictions
from spacepresso.core.records import parse_view
from spacepresso.core.submission import load_submission

__all__ = ["AlignedValidation", "load_test_scores", "load_validation"]

logger = get_logger(__name__)


@dataclass(slots=True)
class AlignedValidation:
    """Validation predictions from every method, on a common index.

    ``scores`` is ``(N, H, W, M)``: image, row, column, method.
    """

    ids: npt.NDArray
    classes: npt.NDArray
    anomaly_types: npt.NDArray
    views: npt.NDArray
    sample_ids: npt.NDArray
    scores: npt.NDArray[np.float32]
    masks: npt.NDArray[np.uint8]
    method_names: list[str]

    @property
    def n_images(self) -> int:
        return int(self.scores.shape[0])

    @property
    def n_methods(self) -> int:
        return int(self.scores.shape[-1])

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.scores.shape[1]), int(self.scores.shape[2])

    def indices_for(self, cls: str) -> npt.NDArray[np.intp]:
        return np.flatnonzero(self.classes == cls)

    def classes_present(self) -> list[str]:
        return sorted(set(self.classes.tolist()))


def load_validation(
    paths: Sequence[Path], method_names: Sequence[str] | None = None
) -> AlignedValidation:
    """Load and align ``local_predictions.npz`` from several runs."""
    if not paths:
        raise ValueError("load_validation() needs at least one predictions file")

    names = list(method_names) if method_names else [Path(p).parent.name for p in paths]
    if len(names) != len(paths):
        raise ValueError(
            f"got {len(paths)} prediction files but {len(names)} method names"
        )

    predictions = [load_predictions(Path(p)) for p in paths]
    for path, prediction in zip(paths, predictions, strict=True):
        for key in ("ids", "scores", "masks", "classes", "anomaly_types"):
            if key not in prediction:
                raise ValueError(f"{path} is missing the {key!r} array")

    common = sorted(set.intersection(*(set(p["ids"].tolist()) for p in predictions)))
    if not common:
        raise ValueError(
            "the methods share no validation image IDs — were they run on "
            "different classes, or with different --only-classes?"
        )
    for name, prediction in zip(names, predictions, strict=True):
        missing = len(prediction["ids"]) - len(common)
        if missing:
            logger.warning(
                "%s has %d validation images the others do not; using the "
                "%d-image intersection",
                name,
                missing,
                len(common),
            )

    height, width = predictions[0]["scores"].shape[1:3]
    logger.info(
        "  aligning %d images × %d methods at %dx%d",
        len(common),
        len(names),
        height,
        width,
    )

    n = len(common)
    scores = np.empty((n, height, width, len(names)), dtype=np.float32)
    masks = np.empty((n, height, width), dtype=np.uint8)
    classes = np.empty(n, dtype=object)
    anomaly_types = np.empty(n, dtype=object)
    views = np.full(n, -1, dtype=np.int32)
    sample_ids = np.empty(n, dtype=object)

    lookups = [
        {image_id: index for index, image_id in enumerate(p["ids"])}
        for p in predictions
    ]

    for position, image_id in enumerate(common):
        for method, (prediction, lookup) in enumerate(
            zip(predictions, lookups, strict=True)
        ):
            row = lookup[image_id]
            score = prediction["scores"][row]
            if score.shape != (height, width):
                score = resize_nearest(score, (height, width)).astype(np.float32)
            scores[position, :, :, method] = score

        reference, lookup = predictions[0], lookups[0]
        row = lookup[image_id]
        classes[position] = str(reference["classes"][row])
        anomaly_types[position] = str(reference["anomaly_types"][row])
        mask = reference["masks"][row]
        if mask.shape != (height, width):
            mask = resize_nearest(mask, (height, width)).astype(np.uint8)
        masks[position] = mask

        sample, view = parse_view(f"{image_id}.png")
        sample_ids[position] = sample
        if view is not None:
            views[position] = view

    found_views = int((views >= 0).sum())
    logger.info("  %d/%d images carry a view index", found_views, n)

    return AlignedValidation(
        ids=np.asarray(common),
        classes=classes.astype(str),
        anomaly_types=anomaly_types.astype(str),
        views=views,
        sample_ids=sample_ids.astype(str),
        scores=scores,
        masks=masks,
        method_names=names,
    )


def load_test_scores(
    submission_paths: Sequence[Path], image_ids: Sequence[str] | None = None
) -> tuple[list[str], npt.NDArray[np.uint8]]:
    """Decode several submissions onto a common set of image IDs."""
    if not submission_paths:
        raise ValueError("load_test_scores() needs at least one submission")

    payloads = [load_submission(Path(p)) for p in submission_paths]
    common = sorted(set.intersection(*(set(p) for p in payloads)))
    if image_ids is not None:
        wanted = set(image_ids)
        common = [i for i in common if i in wanted]
    if not common:
        raise ValueError("the submissions share no image IDs")

    height, width = codec.shape_of(payloads[0][common[0]])
    scores = np.empty((len(common), height, width, len(payloads)), dtype=np.uint8)

    for position, image_id in enumerate(common):
        for method, payload in enumerate(payloads):
            decoded = codec.decode_to_uint8(payload[image_id])
            if decoded.shape != (height, width):
                decoded = resize_nearest(decoded, (height, width)).astype(np.uint8)
            scores[position, :, :, method] = decoded

    logger.info(
        "  decoded %d test images × %d methods at %dx%d (%.1f MB as uint8)",
        len(common),
        len(payloads),
        height,
        width,
        scores.nbytes / 1e6,
    )
    return common, scores
