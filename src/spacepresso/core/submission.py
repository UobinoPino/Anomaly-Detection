"""Reading and writing ``submission.csv``.

One writer (was: 16 copies of ``write_submission``) and one reader (was: 8
copies of ``load_submission``, 4 variants).
"""

from __future__ import annotations

import csv
import sys
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from spacepresso.core import codec
from spacepresso.core.imaging import (
    calibrate_to_unit,
    normalise_to_unit,
    resize_to_submission,
)
from spacepresso.core.logging import get_logger
from spacepresso.core.records import ImageRecord

__all__ = [
    "ScoredRecord",
    "load_submission",
    "load_submission_matrices",
    "write_submission",
    "zip_submission",
]

logger = get_logger(__name__)

#: What a detector hands back: one record and its score map.
ScoredRecord = tuple[ImageRecord, npt.NDArray[np.floating]]

# Submission payloads are long single-line CSV fields; the default limit
# truncates them.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def write_submission(
    scored: Sequence[ScoredRecord],
    out_path: Path,
    *,
    calibrate: bool = True,
    zip_it: bool = True,
) -> Path:
    """Write ``submission.csv`` from scored records.

    Args:
        scored: ``(record, score_map)`` pairs. Maps of any resolution are
            resized to the submission resolution.
        calibrate: map raw scores to [0, 1] using one **global** percentile
            pair across all rows. Detectors produce unbounded scores
            (reconstruction error, negative log-likelihood), and the ordering
            between images is what the pooled metric measures — so the mapping
            has to be global. Pass ``False`` only when scores are already in
            [0, 1], e.g. stacker probabilities.
        zip_it: also write ``<out_path>.zip``, ready to upload.

    Returns:
        The path to the zip when ``zip_it`` is set, otherwise to the CSV.
    """
    if not scored:
        raise ValueError("write_submission() received no scored records")

    # Submission IDs are image stems. A collision does not fail — the CSV just
    # carries two rows with the same ID, and whatever consumes it keeps one.
    # That silently drops predictions on the leaderboard, so it is worth a
    # hard failure here rather than a mystery a week later.
    seen: dict[str, ImageRecord] = {}
    for record, _ in scored:
        if record.stem in seen:
            raise ValueError(
                f"duplicate submission ID {record.stem!r}: "
                f"{seen[record.stem].path} and {record.path}. "
                "Image stems must be unique across the whole test split."
            )
        seen[record.stem] = record

    maps = [resize_to_submission(m) for _r, m in scored]
    if calibrate:
        lo, hi = calibrate_to_unit(maps)
        logger.info("    global score calibration  lo=%.4f  hi=%.4f", lo, hi)
        maps = [normalise_to_unit(m, lo, hi) for m in maps]
    else:
        maps = [np.clip(m, 0.0, 1.0).astype(np.float32) for m in maps]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ID", "Label"])
        for (record, _raw), matrix in zip(scored, maps, strict=True):
            writer.writerow([record.stem, codec.encode_float(matrix)])
    logger.info("    wrote %d rows -> %s", len(scored), out_path)

    if zip_it:
        return zip_submission(out_path)
    return out_path


def zip_submission(csv_path: Path) -> Path:
    """Zip a submission CSV in place, returning the zip path."""
    zip_path = csv_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname=csv_path.name)
    logger.info("    zipped         -> %s", zip_path)
    return zip_path


def _rows_to_dict(rows, path: Path) -> dict[str, str]:
    """Skip the header and collect ``{id: payload}``."""
    if next(rows, None) is None:
        raise ValueError(f"empty submission file: {path}")
    return {row[0]: row[1] for row in rows if len(row) >= 2}


def load_submission(path: Path) -> dict[str, str]:
    """Read a submission CSV (or its zip) into ``{image_id: q8rle payload}``."""
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            inner = next(n for n in archive.namelist() if n.endswith(".csv"))
            text = archive.read(inner).decode("utf-8")
        rows = csv.reader(text.splitlines())
        out = _rows_to_dict(rows, path)
    else:
        with open(path, newline="", encoding="utf-8") as handle:
            out = _rows_to_dict(csv.reader(handle), path)

    if not out:
        raise ValueError(f"submission has a header but no rows: {path}")
    return out


def load_submission_matrices(
    path: Path, ids: Iterable[str] | None = None
) -> dict[str, npt.NDArray[np.uint8]]:
    """Read a submission and decode it to uint8 matrices.

    ``ids`` restricts the decode to a subset, which matters: a full submission
    is ~thousands of 224x224 maps and decoding all of them to float32 is where
    the stacker used to spend its memory budget.
    """
    payloads: Mapping[str, str] = load_submission(path)
    wanted = set(ids) if ids is not None else None
    return {
        key: codec.decode_to_uint8(payload)
        for key, payload in payloads.items()
        if wanted is None or key in wanted
    }
