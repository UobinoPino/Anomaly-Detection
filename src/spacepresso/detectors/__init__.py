"""The detector portfolio.

Every detector is registered by name, so the CLI, the config files and the
stacker all address them the same way::

    from spacepresso.detectors import get_detector, list_detectors

    cls, config_cls = get_detector("patchcore")

Registration is explicit rather than by import-time side effect: the table
below is the list, and a detector that is not in it does not exist. That is
deliberately boring — the previous arrangement, where the set of detectors was
"whichever ``*_baseline.py`` files happen to be in ``models/``", is how
``patchcore_baseline.py`` v1 stayed in the tree long after it was dead.

Detectors are imported lazily. Several pull in heavy optional dependencies
(``transformers`` for WinCLIP and TextAD, ``xgboost`` nowhere here but in the
stacker), and importing the package should not require all of them.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator

from spacepresso.detectors.base import Detector, ScoreMap, ScoreMaps
from spacepresso.runner.config import DetectorConfig

__all__ = [
    "Detector",
    "DetectorConfig",
    "ScoreMap",
    "ScoreMaps",
    "get_detector",
    "list_detectors",
]

#: ``name -> (module, detector class, config class)``
_REGISTRY: dict[str, tuple[str, str, str]] = {
    "patchcore": ("patchcore", "PatchCore", "PatchCoreConfig"),
    "efficientad": ("efficientad", "EfficientAD", "EfficientADConfig"),
    "fastflow": ("fastflow", "FastFlow", "FastFlowConfig"),
    "reverse_distillation": (
        "reverse_distillation",
        "ReverseDistillation",
        "ReverseDistillationConfig",
    ),
    "cfa": ("cfa", "CFA", "CFAConfig"),
    "draem": ("draem", "DRAEM", "DRAEMConfig"),
    "cutpaste": ("cutpaste", "CutPaste", "CutPasteConfig"),
    "glass": ("glass", "GLASS", "GLASSConfig"),
    "uniad": ("uniad", "UniAD", "UniADConfig"),
    "dino_dpmm": ("dino_dpmm", "DinoDPMM", "DinoDPMMConfig"),
    "anomalydino": ("anomalydino", "AnomalyDINO", "AnomalyDINOConfig"),
    "winclip": ("winclip", "WinCLIP", "WinCLIPConfig"),
}


def list_detectors() -> list[str]:
    """Registered detector names, sorted."""
    return sorted(_REGISTRY)


def get_detector(name: str) -> tuple[type[Detector], type[DetectorConfig]]:
    """Resolve a name to its ``(Detector, DetectorConfig)`` pair."""
    try:
        module_name, detector_name, config_name = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown detector {name!r}. Available: {', '.join(list_detectors())}"
        ) from None

    module = importlib.import_module(f"spacepresso.detectors.{module_name}")
    return getattr(module, detector_name), getattr(module, config_name)


def iter_detectors() -> Iterator[tuple[str, type[Detector], type[DetectorConfig]]]:
    """Import and yield every registered detector.

    Used by the architecture test, which checks that each one actually
    imports and satisfies the protocol.
    """
    for name in list_detectors():
        detector, config = get_detector(name)
        yield name, detector, config
