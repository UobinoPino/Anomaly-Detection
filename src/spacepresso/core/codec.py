"""The q8rle submission codec.

Format::

    q8rle <H> <W> <value> <runlength> <value> <runlength> ...

Values are 8-bit (0–255). Runs are laid out in **column-major** order: the
matrix is transposed, flattened, and run-length encoded. An all-constant image
therefore encodes to a single run.

This writes the leaderboard submission, so it has a round-trip test suite.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "MAGIC",
    "Q8RLEError",
    "decode_to_float",
    "decode_to_uint8",
    "encode_float",
    "encode_uint8",
    "shape_of",
]

MAGIC = "q8rle"


class Q8RLEError(ValueError):
    """Raised when a q8rle payload is malformed."""


# ─────────────────────────────────────────────────────────────────────────────
# Encoding
# ─────────────────────────────────────────────────────────────────────────────
def encode_uint8(q: npt.NDArray[np.uint8]) -> str:
    """Run-length encode an already-quantised ``(H, W)`` uint8 matrix."""
    if q.ndim != 2:
        raise Q8RLEError(f"expected a 2-D matrix, got shape {q.shape}")
    h, w = q.shape
    flat = np.ascontiguousarray(q.T).reshape(-1)
    if flat.size == 0:
        return f"{MAGIC} {h} {w}"

    cuts = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.r_[0, cuts]
    lengths = np.r_[cuts, flat.size] - starts

    parts = [MAGIC, str(h), str(w)]
    for value, run in zip(flat[starts], lengths, strict=True):
        parts.append(str(int(value)))
        parts.append(str(int(run)))
    return " ".join(parts)


def encode_float(x: npt.NDArray[np.floating]) -> str:
    """Quantise a ``(H, W)`` float matrix in [0, 1] to 8 bits and encode it.

    Values outside [0, 1] are clipped — the caller is expected to have
    normalised already (see :func:`spacepresso.core.imaging.calibrate_to_unit`).
    """
    q = np.clip(np.rint(np.asarray(x, dtype=np.float32) * 255.0), 0, 255)
    return encode_uint8(q.astype(np.uint8))


# ─────────────────────────────────────────────────────────────────────────────
# Decoding
# ─────────────────────────────────────────────────────────────────────────────
def _parse(payload: str) -> tuple[int, int, npt.NDArray[np.int64]]:
    parts = payload.split()
    if len(parts) < 3:
        raise Q8RLEError(f"truncated payload: {payload[:40]!r}")
    if parts[0] != MAGIC:
        raise Q8RLEError(f"expected magic {MAGIC!r}, got {parts[0]!r}")
    try:
        h, w = int(parts[1]), int(parts[2])
    except ValueError as exc:
        raise Q8RLEError(f"bad dimensions: {parts[1:3]}") from exc
    if h < 0 or w < 0:
        raise Q8RLEError(f"negative dimensions: {h}x{w}")

    body = parts[3:]
    if len(body) % 2:
        raise Q8RLEError(
            f"odd number of body tokens ({len(body)}) — value/run pairs expected"
        )
    return h, w, np.asarray(body, dtype=np.int64) if body else np.empty(0, np.int64)


def shape_of(payload: str) -> tuple[int, int]:
    """Read just the ``(H, W)`` header without decoding the body."""
    h, w, _ = _parse(payload)
    return h, w


def decode_to_uint8(payload: str) -> npt.NDArray[np.uint8]:
    """Decode a q8rle payload to its ``(H, W)`` uint8 matrix."""
    h, w, body = _parse(payload)
    if body.size == 0:
        return np.zeros((h, w), dtype=np.uint8)

    values = body[0::2]
    runs = body[1::2]
    if np.any(values < 0) or np.any(values > 255):
        raise Q8RLEError("value out of the 0–255 range")
    if np.any(runs < 0):
        raise Q8RLEError("negative run length")

    total = int(runs.sum())
    if total != h * w:
        raise Q8RLEError(
            f"run lengths sum to {total}, but the header declares {h}x{w}={h * w}"
        )
    flat = np.repeat(values.astype(np.uint8), runs)
    return flat.reshape(w, h).T


def decode_to_float(payload: str) -> npt.NDArray[np.float32]:
    """Decode a q8rle payload to a ``(H, W)`` float32 matrix in [0, 1]."""
    return decode_to_uint8(payload).astype(np.float32) / 255.0
