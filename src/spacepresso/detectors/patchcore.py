"""PatchCore — nearest-neighbour anomaly detection on a coreset of normal patches.

Roth et al., *Towards Total Recall in Industrial Anomaly Detection* (CVPR 2022).

Training-free in the gradient sense: extract patch features from the
defect-free images, subsample them to a coreset, and score a test patch by its
distance to the nearest bank entry.

Ported from ``models/patchcore_baseline_v2.py`` (1,571 lines). Everything that
was not PatchCore — the dataset scanner, the codec, the metric, the backbone
tables, the run-id hashing, the submission writer, the ablation appender —
moved to :mod:`spacepresso.core` and :mod:`spacepresso.backbones`, which is
what the other 15 modules were importing this file for.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import torch

from spacepresso.backbones import (
    build_backbone,
    patchify_and_combine,
    resolve_target_layer,
    short_tag,
    validate_input_size,
)
from spacepresso.config import DetectorConfig, RuntimeConfig
from spacepresso.core.logging import now_hms
from spacepresso.core.records import ImageRecord
from spacepresso.detectors.base import Detector
from spacepresso.detectors.coreset import MemoryBank, greedy_coreset

__all__ = ["PatchCore", "PatchCoreConfig"]


@dataclass
class PatchCoreConfig(DetectorConfig):
    """PatchCore hyperparameters.

    ``coreset_frac`` is the main quality/cost dial: 10% of patches is the
    paper's setting and what every run in ``baseline_out/`` used.
    """

    backbone: str = "wide_resnet50_2"
    feature_layers: tuple[int, ...] = (2, 3)
    target_layer: int | None = None
    patch_size: int = 3

    coreset_frac: float = 0.10
    coreset_algorithm: str = "minibatch"
    coreset_batch: int = 64
    coreset_fp16: bool = False
    projection_dim: int = 32

    knn_k: int = 1
    memory_dtype: str = "fp16"

    #: Chunk sizes bound peak GPU memory; they do not change the result, so
    #: they are excluded from the fingerprint.
    score_chunk: int = 4096
    memory_chunk: int = 32_768
    project_chunk: int = 65_536
    train_patch_keep: float = 1.0

    _excluded: ClassVar[frozenset[str]] = frozenset(
        {"score_chunk", "memory_chunk", "project_chunk"}
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.feature_layers = tuple(self.feature_layers)
        validate_input_size(self.backbone, self.input_size)
        self.target_layer = resolve_target_layer(
            self.backbone, self.feature_layers, self.target_layer
        )
        if not 0 < self.coreset_frac <= 1:
            raise ValueError(f"coreset_frac must be in (0, 1], got {self.coreset_frac}")

    def fingerprint(self) -> dict[str, object]:
        return {
            key: value
            for key, value in super().fingerprint().items()
            if key not in self._excluded
        }

    def slug_parts(self) -> list[str]:
        layers = "_".join(str(layer) for layer in self.feature_layers)
        parts = [
            short_tag(self.backbone),
            f"L{layers}",
            f"T{self.target_layer}",
            f"in{self.input_size}",
            f"cs{int(self.coreset_frac * 100):02d}",
        ]
        if self.coreset_algorithm == "minibatch":
            parts.append(f"mb{self.coreset_batch}")
        else:
            parts.append("exact")
        if self.knn_k != 1:
            parts.append(f"k{self.knn_k}")
        if self.tta != "none":
            parts.append(f"tta-{self.tta}")
        return parts


class PatchCore(Detector[PatchCoreConfig]):
    """k-NN distance to a coreset of normal patch features."""

    name: ClassVar[str] = "patchcore"
    config_type: ClassVar[type[DetectorConfig]] = PatchCoreConfig

    def __init__(self, config: PatchCoreConfig, runtime: RuntimeConfig) -> None:
        super().__init__(config, runtime)
        self.backbone = build_backbone(config.backbone, device=self.device)
        self.bank: MemoryBank | None = None
        self.grid: tuple[int, int] | None = None

    # ── fit ──────────────────────────────────────────────────────────────
    def fit(self, train_good: Sequence[ImageRecord]) -> None:
        features = self._extract_features(train_good)
        n_select = max(int(self.config.coreset_frac * features.shape[0]), 1)

        self.log.info(
            "    [%s] greedy coreset (%s): selecting %d of %d patches (%.1f%%)",
            now_hms(),
            self.config.coreset_algorithm,
            n_select,
            features.shape[0],
            self.config.coreset_frac * 100,
        )
        indices = greedy_coreset(
            features,
            n_select,
            self.device,
            seed=self.config.seed,
            projection_dim=self.config.projection_dim,
            project_chunk=self.config.project_chunk,
            algorithm=self.config.coreset_algorithm,
            batch_size=self.config.coreset_batch,
        )

        selected = features[indices].to(self.device, non_blocking=True)
        del features
        dtype = torch.float16 if self.config.memory_dtype == "fp16" else torch.float32
        self.bank = MemoryBank(selected, dtype=dtype)
        del selected
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.log.info(
            "    [%s] memory bank ready  shape=(%d, %d)  dtype=%s  (%.1f MB)",
            now_hms(),
            self.bank.size,
            self.bank.dim,
            self.bank.vectors.dtype,
            self.bank.memory_mb(),
        )

    @torch.inference_mode()
    def _extract_features(self, records: Sequence[ImageRecord]) -> torch.Tensor:
        """Patch features for every training image, accumulated on the CPU.

        The bank for one class can be tens of millions of patch vectors, which
        is why this lands on the CPU and only the coreset comes back to the
        GPU.
        """
        self.log.info(
            "    [%s] extracting train/good features (%d images, input=%d, "
            "backbone=%s, layers=%s)",
            now_hms(),
            len(records),
            self.config.input_size,
            self.config.backbone,
            list(self.config.feature_layers),
        )
        loader = self.make_loader(records, batch_size=self.runtime.batch_size)

        keep = self.config.train_patch_keep
        generator = (
            torch.Generator(device=self.device).manual_seed(self.config.seed + 12345)
            if keep < 1.0
            else None
        )

        chunks: list[torch.Tensor] = []
        seen = 0
        for images, _masks, _indices in loader:
            images = images.to(self.device, non_blocking=True)
            patches = self._patch_features(images)

            if self.grid is None:
                side = math.isqrt(patches.shape[1])
                self.grid = (side, side)

            if keep < 1.0:
                total = patches.shape[1]
                n_keep = max(1, int(total * keep))
                chosen = torch.randperm(total, generator=generator, device=self.device)[
                    :n_keep
                ]
                patches = patches[:, chosen, :]

            flat = patches.reshape(-1, patches.shape[-1]).detach()
            if self.config.coreset_fp16:
                flat = flat.half()
            chunks.append(flat.cpu())
            seen += images.shape[0]

        features = torch.cat(chunks, dim=0)
        self.log.info(
            "    -> %d patch features (%s, CPU; %.2f GB)",
            features.shape[0],
            features.dtype,
            features.element_size() * features.numel() / 1e9,
        )
        return features

    def _patch_features(self, images: torch.Tensor) -> torch.Tensor:
        maps = self.backbone(images, layers=self.config.feature_layers)
        assert self.config.target_layer is not None
        return patchify_and_combine(
            maps,
            patch_size=self.config.patch_size,
            target_layer=self.config.target_layer,
        )

    # ── score ────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def score_batch(self, images: torch.Tensor) -> torch.Tensor:
        if self.bank is None:
            raise RuntimeError("PatchCore.score_batch() called before fit()")

        patches = self._patch_features(images)
        batch, n_patches, channels = patches.shape
        side = math.isqrt(n_patches)

        distances = self.bank.distance(
            patches.reshape(-1, channels),
            query_chunk=self.config.score_chunk,
            memory_chunk=self.config.memory_chunk,
            k=self.config.knn_k,
        )
        return self.upsample(distances.reshape(batch, side, side))

    def release(self) -> None:
        self.bank = None
        self.grid = None
        super().release()
