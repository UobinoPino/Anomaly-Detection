"""Builds a tiny synthetic Spacepresso dataset on disk.

Small enough to run a whole experiment in a couple of seconds on CPU, and
structured exactly like the real thing, so the tests exercise the real
``scan_dataset`` / runner / submission path rather than a mock of it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

__all__ = ["build_dataset"]


def _save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def build_dataset(
    root: Path,
    *,
    n_classes: int = 2,
    n_views: int = 2,
    n_good: int = 4,
    n_anomaly: int = 2,
    n_test: int = 3,
    size: int = 64,
    seed: int = 0,
) -> Path:
    """Write a dataset under ``root`` and return it.

    Normal images are smooth gradients with mild noise. Anomalous ones get a
    bright square, and the ground-truth mask marks exactly that square — so a
    working detector scores meaningfully above chance and a broken one does
    not.
    """
    rng = np.random.default_rng(seed)
    root = Path(root)

    for class_index in range(1, n_classes + 1):
        cls = f"class_{class_index:02d}"
        base = np.linspace(40, 200, size, dtype=np.float32)
        canvas = np.outer(base, np.ones(size, dtype=np.float32))

        def normal(offset: float, base: np.ndarray = canvas) -> np.ndarray:
            noisy = base + offset + rng.normal(0, 4, (size, size))
            return np.clip(noisy, 0, 255).astype(np.uint8)

        for index in range(n_good):
            for view in range(n_views):
                _save(
                    root
                    / cls
                    / "train"
                    / "good"
                    / f"{cls}_good{index:02d}_view{view:02d}.png",
                    np.stack([normal(view * 6)] * 3, axis=-1),
                )

        for anomaly_type in ("anomaly_01", "anomaly_02"):
            for index in range(n_anomaly):
                for view in range(n_views):
                    image = normal(view * 6)
                    mask = np.zeros((size, size), dtype=np.uint8)
                    top = rng.integers(8, size - 20)
                    left = rng.integers(8, size - 20)
                    image[top : top + 12, left : left + 12] = 250
                    mask[top : top + 12, left : left + 12] = 255
                    stem = f"{cls}_{anomaly_type}_{index:02d}_view{view:02d}.png"
                    _save(
                        root / cls / "train" / anomaly_type / stem,
                        np.stack([image] * 3, axis=-1),
                    )
                    _save(root / cls / "ground_truth_train" / anomaly_type / stem, mask)

        for index in range(n_test):
            for view in range(n_views):
                image = normal(view * 6)
                if index % 2 == 0:
                    image[20:32, 20:32] = 250
                _save(
                    root / cls / "test" / f"{cls}_test{index:02d}_view{view:02d}.png",
                    np.stack([image] * 3, axis=-1),
                )

    return root
