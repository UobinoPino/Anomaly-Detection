"""Experiment orchestration: the runner and the CLI.

Configuration lives one level up, in :mod:`spacepresso.config`, because the
detectors need it too and nothing in ``detectors`` may depend on ``runner``.
"""

from spacepresso.config import DetectorConfig, RuntimeConfig, resolve_device
from spacepresso.runner.experiment import ClassResult, RunResult, run_experiment

__all__ = [
    "ClassResult",
    "DetectorConfig",
    "RunResult",
    "RuntimeConfig",
    "resolve_device",
    "run_experiment",
]
