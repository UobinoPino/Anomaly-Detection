"""Pre-submission sanity check.

Catches in seconds the kinds of damage that are invisible in a run log and in
the local AP — because they happen *after* the metric is computed, in the
quantisation and encoding step:

* scores outside [0, 1] before quantisation, saturating to 0 or 255;
* maps so dense that the run-length encoding stops compressing and the upload
  balloons;
* one class's scores dominating every other class's, which wrecks the pooled
  ranking even when each class looks fine on its own;
* an implausibly high positive fraction — everything looks anomalous, which
  scores near the base rate.
"""

from __future__ import annotations

import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spacepresso.core import codec
from spacepresso.core.logging import get_logger, setup_logging
from spacepresso.core.records import scan_dataset
from spacepresso.core.submission import load_submission

__all__ = ["Finding", "check_submission"]

logger = get_logger(__name__)

#: Above this fraction of pixels in the top quantisation bucket, the map is
#: almost certainly saturated rather than confident.
SATURATION_LIMIT = 0.02
#: Above this mean positive fraction the submission is calling most of the
#: image anomalous.
DENSITY_LIMIT = 0.25
#: Between-class median ratio beyond which the pooled ranking is dominated by
#: one class.
CLASS_RATIO_LIMIT = 5.0


@dataclass(slots=True)
class Finding:
    level: str
    message: str


def check_submission(
    path: Path, data_root: Path | None = None, sample: int = 400
) -> list[Finding]:
    """Inspect a submission and return what looks wrong.

    ``sample`` bounds how many rows are decoded; the checks are statistical
    and a few hundred images is plenty.
    """
    findings: list[Finding] = []
    payloads = load_submission(Path(path))
    logger.info("  %d rows in %s", len(payloads), path)

    class_of: dict[str, str] = {}
    if data_root is not None:
        class_of = {r.stem: r.cls for r in scan_dataset(data_root)}

    ids = sorted(payloads)
    if len(ids) > sample:
        step = len(ids) // sample
        ids = ids[::step][:sample]

    saturated = 0
    densities: list[float] = []
    per_class: dict[str, list[float]] = {}
    shapes: set[tuple[int, int]] = set()

    for image_id in ids:
        matrix = codec.decode_to_uint8(payloads[image_id])
        shapes.add(matrix.shape)

        top_fraction = float((matrix == 255).mean())
        if top_fraction > SATURATION_LIMIT:
            saturated += 1

        density = float((matrix > 127).mean())
        densities.append(density)
        per_class.setdefault(class_of.get(image_id, "?"), []).append(
            float(np.median(matrix))
        )

    if len(shapes) > 1:
        findings.append(
            Finding("error", f"mixed map shapes in one submission: {sorted(shapes)}")
        )

    if saturated:
        findings.append(
            Finding(
                "warning",
                f"{saturated}/{len(ids)} sampled maps have >{SATURATION_LIMIT:.0%} of "
                "pixels at the maximum — scores were probably clipped before "
                "quantisation, which flattens the top of the ranking",
            )
        )

    mean_density = float(np.mean(densities)) if densities else 0.0
    if mean_density > DENSITY_LIMIT:
        findings.append(
            Finding(
                "warning",
                f"mean positive fraction {mean_density:.1%} — the submission calls "
                "most of the image anomalous, which scores near the base rate",
            )
        )

    if len(per_class) > 1 and "?" not in per_class:
        medians = {cls: float(np.median(values)) for cls, values in per_class.items()}
        highest = max(medians.values())
        lowest = max(min(medians.values()), 1e-6)
        if highest / lowest > CLASS_RATIO_LIMIT:
            worst = max(medians, key=medians.get)  # type: ignore[arg-type]
            best = min(medians, key=medians.get)  # type: ignore[arg-type]
            findings.append(
                Finding(
                    "warning",
                    f"class score scales differ by {highest / lowest:.1f}x "
                    f"({worst} median {highest:.0f} vs {best} {lowest:.0f}) — the "
                    "pooled metric will be dominated by one class; consider a "
                    "per-class rank-normalisation scope",
                )
            )

    csv_path = Path(path)
    if csv_path.suffix == ".csv":
        zip_path = csv_path.with_suffix(".zip")
        if not zip_path.exists():
            findings.append(Finding("info", "no submission.zip beside the CSV"))
        else:
            with zipfile.ZipFile(zip_path) as handle:
                size_mb = sum(i.compress_size for i in handle.infolist()) / 1e6
            logger.info("  zipped size: %.1f MB", size_mb)
            if size_mb > 100:
                findings.append(
                    Finding(
                        "warning",
                        f"the zip is {size_mb:.0f} MB — dense maps compress poorly, "
                        "which usually means the scores are close to uniform",
                    )
                )

    return findings


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--sample", type=int, default=400)
    args = parser.parse_args(argv)

    findings = check_submission(args.submission, args.data_root, args.sample)
    if not findings:
        logger.info("\n  no problems found.")
        return 0

    logger.info("")
    for finding in findings:
        logger.info("  [%s] %s", finding.level.upper(), finding.message)
    return 1 if any(f.level == "error" for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
