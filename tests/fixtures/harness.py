"""Run any detector end-to-end on the synthetic fixture, on CPU.

Used by the smoke tests and during development. Backbone weights are random,
because CI has no network access to ``download.pytorch.org`` — the point is
that the pipeline runs and produces well-formed artifacts, not that the
numbers are good.
"""

from __future__ import annotations

import functools
import tempfile
from pathlib import Path
from typing import Any

from spacepresso.config import RuntimeConfig
from spacepresso.runner.experiment import RunResult, run_experiment

__all__ = ["offline_backbones", "run_detector"]


def offline_backbones() -> None:
    """Patch the backbone factory to build randomly-initialised ResNets."""
    import spacepresso.backbones as backbones
    from spacepresso.backbones.registry import spec_for
    from spacepresso.backbones.resnet import ResNetBackbone

    backbones._build_cached = functools.lru_cache(maxsize=4)(
        lambda name: ResNetBackbone(spec_for(name), pretrained=False)
    )


def run_detector(
    name: str,
    *,
    root: Path | None = None,
    data_root: Path | None = None,
    **config_kwargs: Any,
) -> RunResult:
    """Build the fixture (unless ``data_root`` is given) and run ``name`` on it."""
    from fixtures.dataset import build_dataset
    from spacepresso.detectors import get_detector

    offline_backbones()
    root = root or Path(tempfile.mkdtemp(prefix="spacepresso-test-"))
    data_root = data_root or build_dataset(root / "data")
    report_dir = root / "out"
    report_dir.mkdir(parents=True, exist_ok=True)

    detector_cls, config_cls = get_detector(name)

    runtime = RuntimeConfig(
        data_root=data_root,
        report_dir=report_dir,
        device="cpu",
        num_workers=0,
        batch_size=4,
        score_batch_size=4,
        amp=False,
    )
    defaults: dict[str, Any] = {"input_size": 64, "smooth_sigma": 1.0}
    defaults.update(config_kwargs)
    config = config_cls(**defaults)

    return run_experiment(lambda c, r: detector_cls(c, r), config, runtime)
