"""Per-class spatial priors over where defects occur.

Computed from the ground-truth masks of the local validation split and
consumed by the stacker as the ``s_prior`` feature.

The prior is a real signal on this dataset — parts are photographed in fixed
rigs, so a given class's defects cluster in a characteristic region. It is
also a real risk: a strong prior lets the stacker score by position rather
than by appearance and score well on validation while failing on a test set
whose defects sit elsewhere. Hence the reported ``centre_bias`` figure, which
says how concentrated the prior is and therefore how much to trust it.

Ported from ``exploit_dataset/compute_priors.py``, which wrote a different
file layout than the stacker's loader expected — so the spatial-prior feature
was silently all zeros unless the files were renamed by hand. Both ends now
agree on ``06_heat_<class>.npy``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from spacepresso.core.imaging import gaussian_smooth
from spacepresso.core.logging import get_logger, setup_logging
from spacepresso.core.paths import default_data_root, default_report_dir
from spacepresso.core.records import TRAIN_ANOMALY, ImageRecord, scan_dataset, select
from spacepresso.data.transforms import load_mask

__all__ = ["PRIOR_FILENAME", "compute_priors", "write_priors"]

logger = get_logger(__name__)

#: The name the stacker's loader looks for.
PRIOR_FILENAME = "06_heat_{cls}.npy"


def _centre_bias(heat: npt.NDArray[np.float32]) -> float:
    """Fraction of the prior's mass inside the central 50% x 50% box.

    0.25 means uniform. Much above that means the prior is concentrated, and
    a stacker leaning on it is at risk if the test defects move.
    """
    height, width = heat.shape
    total = float(heat.sum())
    if total <= 0:
        return 0.0
    centre = heat[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
    return float(centre.sum() / total)


def compute_priors(
    records: Sequence[ImageRecord],
    *,
    resolution: int = 128,
    sigma: float = 3.0,
) -> dict[str, dict[str, object]]:
    """Accumulate defect-location heatmaps per class.

    Returns ``{class: {"heat", "n_masks", "centre_bias", "coverage"}}``.
    """
    priors: dict[str, dict[str, object]] = {}
    classes = sorted({r.cls for r in records})

    for cls in classes:
        masked = [
            r for r in select(records, cls=cls, split=TRAIN_ANOMALY) if r.has_mask
        ]
        if not masked:
            logger.warning("%s has no ground-truth masks — skipping its prior", cls)
            continue

        accumulator = np.zeros((resolution, resolution), dtype=np.float64)
        for record in masked:
            mask = load_mask(record.mask_path, resolution)
            accumulator += mask

        heat = (accumulator / len(masked)).astype(np.float32)
        coverage = float(heat.mean())
        heat = gaussian_smooth(heat, sigma)
        peak = float(heat.max())
        if peak > 0:
            heat = (heat / peak).astype(np.float32)

        priors[cls] = {
            "heat": heat,
            "n_masks": len(masked),
            "centre_bias": _centre_bias(heat),
            "coverage": coverage,
        }
        logger.info(
            "  %-12s %4d masks  mean coverage %.2f%%  centre bias %.2f",
            cls,
            len(masked),
            100 * coverage,
            priors[cls]["centre_bias"],
        )

    return priors


def write_priors(priors: dict[str, dict[str, object]], out_dir: Path) -> Path:
    """Write one ``.npy`` per class plus a summary, in the layout the
    stacker's ``--prior-dir`` expects."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, float]] = {}
    for cls, prior in priors.items():
        np.save(out_dir / PRIOR_FILENAME.format(cls=cls), prior["heat"])
        summary[cls] = {
            "n_masks": int(prior["n_masks"]),  # type: ignore[arg-type]
            "centre_bias": round(float(prior["centre_bias"]), 4),  # type: ignore[arg-type]
            "coverage": round(float(prior["coverage"]), 6),  # type: ignore[arg-type]
        }

    summary_path = out_dir / "_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("  wrote %d priors -> %s", len(priors), out_dir)
    return out_dir


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--sigma", type=float, default=3.0)
    args = parser.parse_args(argv)

    data_root = args.data_root or default_data_root()
    out_dir = args.out or (default_report_dir() / "priors")

    records = scan_dataset(data_root)
    priors = compute_priors(records, resolution=args.resolution, sigma=args.sigma)
    if not priors:
        logger.error(
            "no priors computed — are there ground-truth masks under %s?", data_root
        )
        return 1
    write_priors(priors, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
