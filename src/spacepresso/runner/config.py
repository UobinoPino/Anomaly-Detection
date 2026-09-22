"""Run configuration.

There used to be 14 ``RunConfig`` dataclasses and 14 ``make_run_id``
functions, one per detector, each re-declaring ``data_root``, ``report_dir``,
``seed``, ``num_workers``, ``input_size``, ``tta``, ``only_classes``,
``skip_eval``, ``run_tag`` … and each hashing a hand-maintained subset of them
into the run id.

The split here is the one that was implicit in all 14:

* :class:`RuntimeConfig` — where things live and how hard to push the machine.
  Changing any of it must **not** change the result, so none of it enters the
  run-id digest.
* :class:`DetectorConfig` — the science. Every field is part of the digest.

That separation is the whole reason the digest is now trustworthy: previously
``num_workers`` was excluded from some detectors' fingerprints and included in
others', so two runs of the same experiment could land in different
directories.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch

from spacepresso.core.paths import resolve_data_root, resolve_report_dir
from spacepresso.postprocess.tta import TTA_MODES, TTAMode

__all__ = ["DetectorConfig", "RuntimeConfig", "resolve_device"]


def resolve_device(requested: str = "auto") -> torch.device:
    """``auto`` → CUDA when present, else MPS, else CPU."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class RuntimeConfig:
    """Where the run reads and writes, and how hard it pushes the machine.

    Nothing here affects the numbers, so nothing here is hashed into the run
    id — two runs differing only in ``num_workers`` are the same experiment.
    """

    data_root: Path = field(default_factory=lambda: Path("data"))
    report_dir: Path = field(default_factory=lambda: Path("baseline_out"))

    device: str = "auto"
    num_workers: int = 4
    batch_size: int = 32
    score_batch_size: int = 16
    amp: bool = True

    only_classes: tuple[str, ...] = ()
    skip_eval: bool = False
    skip_submission: bool = False
    save_predictions: bool = True
    zip_submission: bool = True
    run_tag: str = ""

    def __post_init__(self) -> None:
        self.data_root = resolve_data_root(self.data_root)
        self.report_dir = resolve_report_dir(self.report_dir)
        if isinstance(self.only_classes, list):
            self.only_classes = tuple(self.only_classes)

    @property
    def torch_device(self) -> torch.device:
        return resolve_device(self.device)


@dataclass
class DetectorConfig:
    """Base for every detector's settings. Subclass it, add fields, done.

    Fields declared here are common to all detectors *and* affect results, so
    they are part of the fingerprint. Subclasses add their own; the default
    :meth:`fingerprint` picks up every field automatically, which is why
    adding a hyperparameter no longer means remembering to add it to a
    hand-written hash dict.
    """

    input_size: int = 224
    seed: int = 0
    tta: TTAMode = "none"
    smooth_sigma: float = 1.5

    def __post_init__(self) -> None:
        if self.tta not in TTA_MODES:
            raise ValueError(f"tta must be one of {TTA_MODES}, got {self.tta!r}")
        if self.input_size <= 0:
            raise ValueError(f"input_size must be positive, got {self.input_size}")

    # ── identity ─────────────────────────────────────────────────────────
    def fingerprint(self) -> dict[str, Any]:
        """The result-determining settings, hashed into the run id.

        Override to exclude a field that provably does not change the output
        (a chunk size, say) — but the default of "everything" is the safe one,
        and was not what the old per-detector hash dicts did.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def slug_parts(self) -> list[str]:
        """Human-readable fragments for the run directory name.

        The base contributes resolution and TTA; subclasses prepend their own
        (backbone, iterations, and so on).
        """
        parts = [f"in{self.input_size}"]
        if self.tta != "none":
            parts.append(f"tta-{self.tta}")
        return parts

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe rendering for ``config.json``."""
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }
