"""CutPaste — self-supervised pretext task, then nearest-neighbour scoring.

Li et al., *CutPaste: Self-Supervised Learning for Anomaly Detection and
Localization* (CVPR 2021).

Two stages:

1. **Pretext.** Train a classifier to tell normal images from two kinds of
   corruption — a pasted rectangle and a pasted thin "scar". Neither looks
   like a real defect, but learning to spot the seam forces the encoder to
   represent local texture continuity, which is what real defects break.
2. **Scoring.** Discard the classifier head and use the fine-tuned encoder as
   a feature extractor, then score exactly as PatchCore does: distance to a
   coreset bank of normal patches.

The original reimplemented the memory bank and the chunked nearest-neighbour
search; both now come from :mod:`spacepresso.detectors.coreset`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from spacepresso.backbones import (
    build_backbone,
    patchify_and_combine,
    resolve_target_layer,
    short_tag,
    validate_input_size,
)
from spacepresso.backbones.registry import family_of
from spacepresso.config import DetectorConfig, RuntimeConfig
from spacepresso.core.logging import now_hms
from spacepresso.core.records import ImageRecord
from spacepresso.data.synthesis import cutpaste, cutpaste_scar
from spacepresso.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from spacepresso.detectors.base import Detector
from spacepresso.detectors.coreset import MemoryBank, greedy_coreset
from spacepresso.detectors.training import train_loop

__all__ = ["CutPaste", "CutPasteConfig"]

#: normal / patch / scar
N_CLASSES = 3


@dataclass
class CutPasteConfig(DetectorConfig):
    backbone: str = "resnet18"
    feature_layers: tuple[int, ...] = (2, 3)
    target_layer: int | None = None
    patch_size: int = 3
    projection_dim: int = 256

    area_ratio: tuple[float, float] = (0.02, 0.15)
    aspect_ratio: tuple[float, float] = (0.3, 3.3)
    colour_jitter: float = 0.1
    scar_width: tuple[int, int] = (10, 25)
    scar_height: tuple[int, int] = (2, 16)

    coreset_frac: float = 0.10
    coreset_algorithm: str = "minibatch"
    coreset_batch: int = 64
    memory_dtype: str = "fp16"

    total_iters: int | None = 2500
    epochs: int = 100
    train_batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-5

    score_chunk: int = 4096
    memory_chunk: int = 32_768

    _excluded: ClassVar[frozenset[str]] = frozenset({"score_chunk", "memory_chunk"})

    def __post_init__(self) -> None:
        super().__post_init__()
        self.feature_layers = tuple(self.feature_layers)
        validate_input_size(self.backbone, self.input_size)
        self.target_layer = resolve_target_layer(
            self.backbone, self.feature_layers, self.target_layer
        )

    def fingerprint(self) -> dict[str, object]:
        return {
            k: v for k, v in super().fingerprint().items() if k not in self._excluded
        }

    def slug_parts(self) -> list[str]:
        parts = [
            short_tag(self.backbone),
            f"in{self.input_size}",
            f"it{self.total_iters}" if self.total_iters else f"e{self.epochs}",
            f"bs{self.train_batch_size}",
            f"cs{int(self.coreset_frac * 100):02d}",
        ]
        if self.tta != "none":
            parts.append(f"tta-{self.tta}")
        return parts


class _PretextDataset(Dataset):
    """Yields one image and its label: 0 normal, 1 cut-paste, 2 scar."""

    def __init__(self, records: Sequence[ImageRecord], config: CutPasteConfig):
        self.records = list(records)
        self.config = config
        self.mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
        self.std = np.asarray(IMAGENET_STD, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        size = self.config.input_size
        with Image.open(self.records[index].path) as handle:
            image = np.asarray(
                handle.convert("RGB").resize((size, size), Image.BILINEAR)
            )

        worker = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(
            (
                self.config.seed,
                worker.id if worker is not None else 0,
                index,
                int(torch.randint(0, 2**31, (1,)).item()),
            )
        )

        label = int(rng.integers(0, N_CLASSES))
        if label == 1:
            image = cutpaste(
                image,
                rng,
                area_ratio=self.config.area_ratio,
                aspect_ratio=self.config.aspect_ratio,
                jitter=self.config.colour_jitter,
            )
        elif label == 2:
            image = cutpaste_scar(
                image,
                rng,
                width_range=self.config.scar_width,
                height_range=self.config.scar_height,
                jitter=self.config.colour_jitter,
            )

        normalised = (image.astype(np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(normalised.transpose(2, 0, 1).copy()), label


class _Classifier(nn.Module):
    """Global-pooled backbone features plus a small classification head.

    The head is thrown away after the pretext task; only the encoder matters.
    """

    def __init__(self, backbone, layers: Sequence[int], projection_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.layers = tuple(layers)
        in_dim = backbone.total_channels(self.layers)
        self.head = nn.Sequential(
            nn.Linear(in_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maps = self.backbone(x, layers=self.layers)
        pooled = [
            maps[layer].float().clone().mean(dim=(-2, -1)) for layer in self.layers
        ]
        return self.head(torch.cat(pooled, dim=1))


class CutPaste(Detector[CutPasteConfig]):
    name: ClassVar[str] = "cutpaste"
    config_type: ClassVar[type[DetectorConfig]] = CutPasteConfig

    def __init__(self, config: CutPasteConfig, runtime: RuntimeConfig) -> None:
        super().__init__(config, runtime)
        self.backbone = build_backbone(config.backbone, device=self.device)
        self.classifier: _Classifier | None = None
        self.bank: MemoryBank | None = None

    def fit(self, train_good: Sequence[ImageRecord]) -> None:
        self._train_pretext(train_good)
        self._fit_memory_bank(train_good)

    def _train_pretext(self, records: Sequence[ImageRecord]) -> None:
        """Fine-tune the encoder on the three-way corruption task.

        Only ResNet backbones are unfrozen. A ViT fine-tuned on a few hundred
        images of one class overfits the pretext task within a few hundred
        steps and the features get *worse*, so there the backbone stays frozen
        and only the head trains — which still shapes nothing, so we skip the
        stage entirely and fall through to nearest-neighbour scoring on the
        pretrained features.
        """
        if family_of(self.config.backbone) != "resnet":
            self.log.info(
                "    %s is a transformer backbone; skipping the pretext stage "
                "and scoring on frozen features",
                self.config.backbone,
            )
            return

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(True)
        self.backbone.train()

        self.classifier = _Classifier(
            self.backbone, self.config.feature_layers, self.config.projection_dim
        ).to(self.device)

        dataset = _PretextDataset(records, self.config)
        loader = self.make_loader(
            records,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            drop_last=True,
            dataset=dataset,
        )
        self._labels: torch.Tensor | None = None

        def loss_fn(images: torch.Tensor) -> torch.Tensor:
            assert self.classifier is not None and self._labels is not None
            logits = self.classifier(images)
            return nn.functional.cross_entropy(
                logits, self._labels.to(self.device, non_blocking=True)
            )

        train_loop(
            [self.classifier],
            _LabelCapture(loader, self),
            loss_fn,
            device=self.device,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            total_iters=self.config.total_iters,
            epochs=self.config.epochs,
            amp=self.runtime.amp,
            label="cutpaste-pretext",
        )
        self.backbone.freeze()

    def _patch_features(self, images: torch.Tensor) -> torch.Tensor:
        maps = self.backbone(images, layers=self.config.feature_layers)
        assert self.config.target_layer is not None
        return patchify_and_combine(
            maps,
            patch_size=self.config.patch_size,
            target_layer=self.config.target_layer,
        )

    @torch.inference_mode()
    def _fit_memory_bank(self, records: Sequence[ImageRecord]) -> None:
        self.log.info(
            "    [%s] extracting train/good patch features (%d images)",
            now_hms(),
            len(records),
        )
        chunks = [
            self._patch_features(images.to(self.device, non_blocking=True))
            .reshape(-1, self.backbone.total_channels(self.config.feature_layers))
            .cpu()
            for images, _masks, _indices in self.make_loader(
                records, batch_size=self.runtime.batch_size
            )
        ]
        features = torch.cat(chunks, dim=0)

        n_select = max(int(self.config.coreset_frac * features.shape[0]), 1)
        self.log.info(
            "    [%s] coreset: %d of %d", now_hms(), n_select, features.shape[0]
        )
        indices = greedy_coreset(
            features,
            n_select,
            self.device,
            seed=self.config.seed,
            algorithm=self.config.coreset_algorithm,
            batch_size=self.config.coreset_batch,
        )
        dtype = torch.float16 if self.config.memory_dtype == "fp16" else torch.float32
        self.bank = MemoryBank(features[indices].to(self.device), dtype=dtype)
        self.log.info(
            "    [%s] memory bank (%d, %d)", now_hms(), self.bank.size, self.bank.dim
        )

    @torch.inference_mode()
    def score_batch(self, images: torch.Tensor) -> torch.Tensor:
        if self.bank is None:
            raise RuntimeError("CutPaste.score_batch() called before fit()")
        features = self._patch_features(images)
        batch, n_patches, channels = features.shape
        side = math.isqrt(n_patches)
        distances = self.bank.distance(
            features.reshape(-1, channels),
            query_chunk=self.config.score_chunk,
            memory_chunk=self.config.memory_chunk,
        )
        return self.upsample(distances.reshape(batch, side, side))

    def release(self) -> None:
        self.classifier = None
        self.bank = None
        # The encoder was fine-tuned for this class; rebuild it so the next
        # class starts from pretrained weights rather than inheriting them.
        self.backbone = build_backbone(
            self.config.backbone, device=self.device, cache=False
        )
        super().release()


class _LabelCapture:
    """Stashes the pretext labels so the shared training loop stays generic."""

    def __init__(self, loader, detector: CutPaste) -> None:
        self._loader = loader
        self._detector = detector

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self):
        for images, labels in self._loader:
            self._detector._labels = labels
            yield images
