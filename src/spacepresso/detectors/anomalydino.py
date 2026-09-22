"""AnomalyDINO — training-free nearest-neighbour matching on DINO features.

Damm et al., *AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection
with DINOv2* (WACV 2025).

The simplest thing that works: take DINO patch tokens from one block, keep the
foreground ones, and score a test patch by its cosine distance to the nearest
normal patch. No training at all — ``fit`` is feature extraction.

The foreground mask is what separates this from a plain PatchCore on DINO
features. Background patches are numerous and nearly identical, so they
dominate the bank while carrying no information; worse, a defect that happens
to look like background scores as normal. The first principal component of the
patch distribution separates object from background almost perfectly on this
dataset, so the bank keeps only patches in the tails of that projection.

Using ``|PC1|`` rather than a signed threshold is deliberate: the sign of a
principal component is arbitrary, so thresholding on the signed value would
keep the object on some classes and the background on others.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F

from spacepresso.backbones import build_backbone, short_tag, validate_input_size
from spacepresso.backbones.registry import patch_size_of
from spacepresso.core.logging import now_hms
from spacepresso.core.records import ImageRecord
from spacepresso.detectors.base import Detector
from spacepresso.detectors.coreset import MemoryBank
from spacepresso.runner.config import DetectorConfig, RuntimeConfig

__all__ = ["AnomalyDINO", "AnomalyDINOConfig"]


@dataclass
class AnomalyDINOConfig(DetectorConfig):
    backbone: str = "dinov2_vits14_reg"
    #: Which transformer block to read. Later blocks are more semantic; the
    #: last one is the paper's choice.
    block: int = 11

    knn_k: int = 1
    use_foreground_mask: bool = True
    foreground_keep_pct: float = 75.0
    bank_dtype: str = "fp16"

    query_chunk: int = 4096
    memory_chunk: int = 32_768

    _excluded: ClassVar[frozenset[str]] = frozenset({"query_chunk", "memory_chunk"})

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_input_size(self.backbone, self.input_size)
        if not 0 < self.foreground_keep_pct <= 100:
            raise ValueError(
                f"foreground_keep_pct must be in (0, 100], got "
                f"{self.foreground_keep_pct}"
            )

    def fingerprint(self) -> dict[str, object]:
        return {
            k: v for k, v in super().fingerprint().items() if k not in self._excluded
        }

    def slug_parts(self) -> list[str]:
        parts = [
            short_tag(self.backbone),
            f"b{self.block}",
            f"in{self.input_size}",
            f"k{self.knn_k}",
        ]
        if self.use_foreground_mask:
            parts.append(f"fg{self.foreground_keep_pct:g}")
        if self.tta != "none":
            parts.append(f"tta-{self.tta}")
        return parts


def foreground_pca(
    patches: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and first principal component of a patch bank."""
    mean = patches.mean(dim=0)
    centred = patches - mean
    _u, _s, v = torch.svd_lowrank(centred, q=2, niter=4)
    return mean, v[:, 0].contiguous()


def project_onto_pc1(
    patches: torch.Tensor, mean: torch.Tensor, component: torch.Tensor
) -> torch.Tensor:
    return (patches - mean) @ component


class AnomalyDINO(Detector[AnomalyDINOConfig]):
    name: ClassVar[str] = "anomalydino"
    config_type: ClassVar[type[DetectorConfig]] = AnomalyDINOConfig

    def __init__(self, config: AnomalyDINOConfig, runtime: RuntimeConfig) -> None:
        super().__init__(config, runtime)
        self.backbone = build_backbone(config.backbone, device=self.device)
        self.bank: MemoryBank | None = None

    def _tokens(self, images: torch.Tensor) -> torch.Tensor:
        """``(B, P, C)`` patch tokens from the configured block."""
        feature = self.backbone(images, layers=(self.config.block,))[self.config.block]
        batch, channels, height, width = feature.shape
        return (
            feature.permute(0, 2, 3, 1).reshape(batch, height * width, channels).float()
        )

    def _grid(self) -> tuple[int, int]:
        patch = patch_size_of(self.config.backbone)
        assert patch is not None
        side = self.config.input_size // patch
        return side, side

    # ── fit ──────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def fit(self, train_good: Sequence[ImageRecord]) -> None:
        chunks = []
        for images, _masks, _indices in self.make_loader(
            train_good, batch_size=self.runtime.batch_size
        ):
            tokens = self._tokens(images.to(self.device, non_blocking=True))
            chunks.append(tokens.reshape(-1, tokens.shape[-1]).cpu())

        patches = torch.cat(chunks, dim=0).to(self.device)
        self.log.info(
            "    [%s] extracted %d patches from %d images (C=%d)",
            now_hms(),
            patches.shape[0],
            len(train_good),
            patches.shape[1],
        )

        if self.config.use_foreground_mask:
            patches = self._keep_foreground(patches)
        else:
            self.log.info("    foreground mask disabled — bank size %d", patches.shape[0])

        dtype = torch.float16 if self.config.bank_dtype == "fp16" else torch.float32
        self.bank = MemoryBank(patches, dtype=dtype)
        self.log.info(
            "    bank ready (%d, %d) %s — %.1f MB",
            self.bank.size,
            self.bank.dim,
            self.bank.vectors.dtype,
            self.bank.memory_mb(),
        )

    def _keep_foreground(self, patches: torch.Tensor) -> torch.Tensor:
        mean, component = foreground_pca(patches)
        projection = project_onto_pc1(patches, mean, component).abs()
        threshold = float(
            torch.quantile(
                projection.float(), 1.0 - self.config.foreground_keep_pct / 100.0
            )
        )
        keep = projection >= threshold
        self.log.info(
            "    foreground mask: keep=%.1f%%  |PC1| threshold=%.4f  keeping %d/%d",
            self.config.foreground_keep_pct,
            threshold,
            int(keep.sum()),
            patches.shape[0],
        )
        return patches[keep]

    # ── score ────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def score_batch(self, images: torch.Tensor) -> torch.Tensor:
        if self.bank is None:
            raise RuntimeError("AnomalyDINO.score_batch() called before fit()")

        tokens = self._tokens(images)
        batch, n_patches, channels = tokens.shape
        distances = self.bank.distance(
            F.normalize(tokens.reshape(-1, channels), p=2, dim=-1),
            query_chunk=self.config.query_chunk,
            memory_chunk=self.config.memory_chunk,
            k=self.config.knn_k,
        )
        side = int(math.isqrt(n_patches))
        return self.upsample(distances.reshape(batch, side, side))

    def release(self) -> None:
        self.bank = None
        super().release()
