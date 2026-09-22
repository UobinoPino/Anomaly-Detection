"""Writing raw per-pixel predictions for the stacker to consume.

Replaces ``models/local_preds_saver.py`` and ``models/test_preds_saver.py``,
which were two near-identical accumulators — and, incidentally, a module named
``test_*`` that pytest would try to import as a test.

Both produced the same npz layout and both held **every score map and every
mask in RAM as Python lists** before calling ``np.stack``, so peak memory was
twice the size of the finished array. At 224x224 float32 over a few thousand
validation views that is several gigabytes of avoidable pressure during the
most memory-hungry phase of a run. This writer streams each map to a raw
append-only file and memory-maps it at save time, so peak RSS is one score map
regardless of dataset size.

Output keys (unchanged, so existing ``.npz`` files still load):

``local_predictions.npz``
    ``ids, classes, anomaly_types, image_paths, scores, masks, score_range``
``test_predictions.npz``
    ``ids, classes, views, image_paths, scores``
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType

import numpy as np
import numpy.typing as npt

from spacepresso.core.imaging import resize_nearest
from spacepresso.core.logging import get_logger
from spacepresso.core.records import ImageRecord

__all__ = ["PredictionWriter", "load_predictions", "write_test_predictions"]

logger = get_logger(__name__)


def _as_2d(array: object, dtype: npt.DTypeLike) -> npt.NDArray:
    """Coerce a torch tensor or ndarray to a 2-D NumPy array of ``dtype``."""
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    out = np.asarray(array, dtype=dtype)
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]
    if out.ndim != 2:
        raise ValueError(f"expected a 2-D map, got shape {out.shape}")
    return out


class PredictionWriter:
    """Streaming accumulator for per-pixel predictions.

    Use as a context manager, or call :meth:`save` explicitly::

        with PredictionWriter(with_masks=True) as writer:
            for record, score, mask in ...:
                writer.add(record, score, mask)
            writer.save(run_dir / "local_predictions.npz")

    Score values are stored **as-is**: no clipping, no rescaling. Density and
    reconstruction detectors produce unbounded scores, and the upper tail is
    exactly the anomaly signal — normalising here would destroy what the
    stacker is being given the raw maps for.
    """

    def __init__(self, *, with_masks: bool, scratch_dir: Path | None = None) -> None:
        self.with_masks = with_masks
        self._scratch = Path(
            scratch_dir or tempfile.mkdtemp(prefix="spacepresso-preds-")
        )
        self._scratch.mkdir(parents=True, exist_ok=True)

        self._shape: tuple[int, int] | None = None
        self._n = 0
        self._ids: list[str] = []
        self._classes: list[str] = []
        self._anomaly_types: list[str] = []
        self._views: list[int] = []
        self._image_paths: list[str] = []

        # These stay open for the writer's lifetime by design — it is a
        # streaming accumulator, and is itself a context manager. noqa: the
        # rule cannot see that ownership is transferred to the object.
        self._score_fh = open(self._scratch / "scores.raw", "wb")  # noqa: SIM115
        self._mask_fh = (
            open(self._scratch / "masks.raw", "wb")  # noqa: SIM115
            if with_masks
            else None
        )

        self._n_nonfinite = 0
        self._n_constant = 0
        self._min = float("inf")
        self._max = float("-inf")

    # ── lifecycle ────────────────────────────────────────────────────────
    def __enter__(self) -> PredictionWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if not self._score_fh.closed:
            self._score_fh.close()
        if self._mask_fh is not None and not self._mask_fh.closed:
            self._mask_fh.close()

    def __len__(self) -> int:
        return self._n

    # ── accumulation ─────────────────────────────────────────────────────
    def add(
        self,
        record: ImageRecord,
        score_map: object,
        mask: object | None = None,
    ) -> None:
        """Append one image's prediction.

        The first call fixes the output resolution; later maps of a different
        shape are nearest-neighbour resized onto it, with a warning. The old
        implementation deferred that to ``save()``, which meant the mismatch
        surfaced after the whole run rather than on the image that caused it.
        """
        score = _as_2d(score_map, np.float32)
        if self._shape is None:
            self._shape = score.shape
        elif score.shape != self._shape:
            logger.warning(
                "prediction shape %s != %s for %s — nearest-neighbour resizing; "
                "this usually means a detector changed resolution mid-run",
                score.shape,
                self._shape,
                record.stem,
            )
            score = resize_nearest(score, self._shape).astype(np.float32)

        score = self._sanitise(score)
        self._score_fh.write(np.ascontiguousarray(score, np.float32).tobytes())

        if self.with_masks:
            if mask is None:
                raise ValueError(
                    f"{record.stem}: writer was built with_masks=True but no mask "
                    "was supplied"
                )
            m = _as_2d(mask, np.uint8)
            if m.shape != self._shape:
                m = resize_nearest(m, self._shape).astype(np.uint8)
            m = (m > 0).astype(np.uint8)
            assert self._mask_fh is not None
            self._mask_fh.write(np.ascontiguousarray(m).tobytes())

        self._ids.append(record.stem)
        self._classes.append(record.cls)
        self._anomaly_types.append(record.anomaly_type or "unknown")
        self._views.append(record.view if record.view is not None else -1)
        self._image_paths.append(str(record.path))
        self._n += 1

    def _sanitise(self, score: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Replace NaN/inf with the image's finite min/max.

        Rank-based AP only cares about ordering among finite values, so this
        preserves the metric while keeping the array stackable downstream.
        """
        finite = np.isfinite(score)
        if not finite.all():
            self._n_nonfinite += 1
            if finite.any():
                lo, hi = float(score[finite].min()), float(score[finite].max())
                score = np.nan_to_num(score, nan=lo, posinf=hi, neginf=lo)
            else:
                score = np.zeros_like(score)

        lo, hi = float(score.min()), float(score.max())
        if lo == hi:
            self._n_constant += 1
        self._min = min(self._min, lo)
        self._max = max(self._max, hi)
        return score

    # ── output ───────────────────────────────────────────────────────────
    def save(self, path: Path) -> Path:
        """Write the npz and report anything that looked wrong."""
        if self._n == 0:
            raise RuntimeError("PredictionWriter.save(): nothing accumulated")
        assert self._shape is not None
        self.close()

        h, w = self._shape
        scores = np.memmap(
            self._scratch / "scores.raw",
            dtype=np.float32,
            mode="r",
            shape=(self._n, h, w),
        )

        arrays: dict[str, npt.NDArray] = {
            "ids": np.asarray(self._ids, dtype=object),
            "classes": np.asarray(self._classes, dtype=object),
            "image_paths": np.asarray(self._image_paths, dtype=object),
            "scores": scores,
        }
        if self.with_masks:
            arrays["masks"] = np.memmap(
                self._scratch / "masks.raw",
                dtype=np.uint8,
                mode="r",
                shape=(self._n, h, w),
            )
            arrays["anomaly_types"] = np.asarray(self._anomaly_types, dtype=object)
            arrays["score_range"] = np.asarray([self._min, self._max], np.float32)
        else:
            arrays["views"] = np.asarray(self._views, dtype=np.int32)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # savez_compressed's first positional is the file; the rest are the
        # named arrays. mypy reads the stub's second parameter as the
        # `allow_pickle` flag, hence the cast.
        np.savez_compressed(path, **arrays)  # type: ignore[arg-type]

        self._report(path, scores)
        return path

    def _report(self, path: Path, scores: npt.NDArray) -> None:
        logger.info(
            "    saved %d predictions (%dx%d) -> %s",
            self._n,
            scores.shape[1],
            scores.shape[2],
            path,
        )
        logger.info(
            "      score range [%.4g, %.4g] (stored as-is, no clipping)",
            self._min,
            self._max,
        )
        if self._n_nonfinite:
            logger.warning(
                "%d/%d images contained NaN/inf — replaced with per-image finite "
                "min/max. Investigate the detector.",
                self._n_nonfinite,
                self._n,
            )
        if self._n_constant:
            logger.warning(
                "%d/%d images have a constant score map; their AP is "
                "ill-defined and the detector is degenerate on those samples.",
                self._n_constant,
                self._n,
            )
        # Tail-saturation heuristic: a flat top 0.1% means the score was
        # clipped upstream, which is the single most common integration bug.
        top = float(
            np.percentile(np.asarray(scores[:: max(1, self._n // 64) or 1]), 99.9)
        )
        peak = float(self._max)
        if peak > 0 and np.isclose(top, peak):
            logger.warning(
                "the top 0.1%% of pixels are tied at the global max (%.4g) — the "
                "score looks clipped before reaching the writer. Pass the RAW "
                "detector score, not a calibrated one.",
                peak,
            )


def write_test_predictions(
    scored: Sequence[tuple[ImageRecord, object]], run_dir: Path
) -> Path:
    """Convenience wrapper for the test split (no masks)."""
    with PredictionWriter(with_masks=False) as writer:
        for record, score in scored:
            writer.add(record, score)
        return writer.save(Path(run_dir) / "test_predictions.npz")


def load_predictions(path: Path) -> dict[str, npt.NDArray]:
    """Load a predictions npz into a plain dict of arrays."""
    with np.load(path, allow_pickle=True) as handle:
        return {key: handle[key] for key in handle.files}
