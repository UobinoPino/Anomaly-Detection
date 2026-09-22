"""The detector interface.

A detector is an **algorithm**, not a program. It answers two questions —
"what does normal look like?" (:meth:`Detector.fit`) and "how anomalous is
this?" (:meth:`Detector.score`) — and nothing else. It does not parse
arguments, hash run ids, write CSVs, compute AP, print progress banners, or
decide which classes to run.

Ground-truth masks are deliberately absent from this interface. A detector has
no business reading the labels it is being evaluated against; the runner loads
them and computes the metric.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar, Generic, TypeVar

import numpy as np
import numpy.typing as npt
import torch

from spacepresso.config import DetectorConfig, RuntimeConfig
from spacepresso.core.logging import get_logger
from spacepresso.core.records import ImageRecord
from spacepresso.data.datasets import make_loader
from spacepresso.postprocess.tta import apply_tta

__all__ = ["Detector", "ScoreMap", "ScoreMaps"]

#: One ``(H, W)`` anomaly score map. Higher means more anomalous. The scale is
#: the detector's own — unbounded is fine and expected.
ScoreMap = npt.NDArray[np.float32]

#: One score map per input record, in the same order.
ScoreMaps = list[ScoreMap]

ConfigT = TypeVar("ConfigT", bound=DetectorConfig)


class Detector(ABC, Generic[ConfigT]):
    """Base class for every anomaly detector in the portfolio.

    Lifecycle, per class, driven by the runner::

        detector = PatchCore(config, runtime)
        detector.fit(train_good_records)
        maps = detector.score(validation_records)
        maps = detector.score(test_records)
        detector.release()

    Subclasses implement :meth:`fit` and one of :meth:`score` or
    :meth:`score_batch`. Implementing ``score_batch`` — "score one batch of
    images, no augmentation" — is the usual choice: the default :meth:`score`
    then supplies batching, TTA and device movement.
    """

    #: Stable identifier, used in run ids, config files and ablation rows.
    name: ClassVar[str]

    #: The config dataclass this detector expects. The CLI reads it to build
    #: its arguments, so a detector's flags are derived from its config rather
    #: than hand-written twice.
    config_type: ClassVar[type[DetectorConfig]]

    def __init__(self, config: ConfigT, runtime: RuntimeConfig) -> None:
        self.config = config
        self.runtime = runtime
        self.device = runtime.torch_device
        self.log = get_logger(f"spacepresso.detectors.{self.name}")

    # ── required ─────────────────────────────────────────────────────────
    @abstractmethod
    def fit(self, train_good: Sequence[ImageRecord]) -> None:
        """Learn what normal looks like, from defect-free images only.

        Called once per class. Implementations that train a network should
        keep the trained state on ``self``; :meth:`release` frees it.
        """
        raise NotImplementedError

    # ── one of these two ─────────────────────────────────────────────────
    def score_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Score one un-augmented batch: ``(B, C, H, W)`` → ``(B, H, W)``.

        The typical thing to implement. TTA, batching and device placement are
        handled by :meth:`score`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement score_batch() or override score()"
        )

    def score(self, records: Sequence[ImageRecord]) -> ScoreMaps:
        """Score records, returning one map per record in the same order.

        Override only when the detector cannot work image-by-image — the
        multi-view consensus path, for instance, needs all views of a sample
        together. Otherwise implement :meth:`score_batch` and leave this
        alone.
        """
        if not records:
            return []

        loader = self.make_loader(records)
        maps: ScoreMaps = [None] * len(records)  # type: ignore[list-item]

        for images, _masks, indices in loader:
            images = images.to(self.device, non_blocking=True)
            scored = apply_tta(self.score_batch, images, self.config.tta)
            scored = scored.detach().float().cpu().numpy()
            for position, index in enumerate(indices.tolist()):
                maps[index] = scored[position].astype(np.float32)

        missing = [i for i, m in enumerate(maps) if m is None]
        if missing:
            raise RuntimeError(
                f"{self.name}: {len(missing)} records were never scored "
                f"(first: {records[missing[0]].stem})"
            )
        return maps

    # ── optional hooks ───────────────────────────────────────────────────
    def release(self) -> None:
        """Drop per-class state between classes."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def fingerprint(self) -> dict[str, object]:
        """Result-determining settings, for the run-id digest."""
        return {"detector": self.name, **self.config.fingerprint()}

    def slug(self) -> str:
        """Human-readable run-directory fragment, e.g. ``effad_dnv2b14r_in392``."""
        return "_".join([self.name, *self.config.slug_parts()])

    # ── helpers for subclasses ───────────────────────────────────────────
    def make_loader(
        self,
        records: Sequence[ImageRecord],
        *,
        batch_size: int | None = None,
        shuffle: bool = False,
        drop_last: bool = False,
        dataset: object | None = None,
    ):
        """Dataloader honouring the run's worker, batch and seeding policy."""
        return make_loader(
            records,
            batch_size=batch_size or self.runtime.score_batch_size,
            input_size=self.config.input_size,
            num_workers=self.runtime.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            dataset=dataset,  # type: ignore[arg-type]
        )

    def autocast(self):
        """AMP context, enabled only on CUDA and only when the run asks."""
        enabled = self.runtime.amp and self.device.type == "cuda"
        return torch.amp.autocast("cuda", enabled=enabled)

    def upsample(self, maps: torch.Tensor) -> torch.Tensor:
        """Resize ``(B, H', W')`` feature-grid scores to the input resolution."""
        size = self.config.input_size
        if maps.shape[-2:] == (size, size):
            return maps
        return torch.nn.functional.interpolate(
            maps.unsqueeze(1), size=(size, size), mode="bilinear", align_corners=False
        ).squeeze(1)
