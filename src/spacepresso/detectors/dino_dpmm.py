"""DINO-DPMM — a Dirichlet-process Gaussian mixture over DINO patch features.

A density model rather than a distance model: fit a mixture of Gaussians to
the normal patch features and score by negative log-likelihood. Where
PatchCore asks "how far is the nearest normal patch?", this asks "how likely
is this patch under the distribution of normal patches?" — which handles
multi-modal normality (several legitimate appearances of the same part)
without needing an example of each mode in the bank.

The Dirichlet-process prior means the number of mixture components is
inferred rather than chosen: fit with a generous maximum and a
concentration prior below 1, and the model prunes itself to the components the
data supports.

PCA first, for two reasons: a 768-dimensional full-covariance mixture has more
parameters than there are patches to fit them, and the EM cost is quadratic in
dimension.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

from spacepresso.backbones import build_backbone, short_tag, validate_input_size
from spacepresso.backbones.registry import patch_size_of
from spacepresso.core.logging import now_hms
from spacepresso.core.records import ImageRecord
from spacepresso.detectors.base import Detector
from spacepresso.runner.config import DetectorConfig, RuntimeConfig

__all__ = ["DinoDPMM", "DinoDPMMConfig", "TorchPCA"]


@dataclass
class DinoDPMMConfig(DetectorConfig):
    backbone: str = "dinov2_vitb14_reg"
    feature_layers: tuple[int, ...] = (9,)

    pca_dim: int = 64
    pca_fit_subsample: int = 50_000

    max_components: int = 30
    covariance_type: str = "diag"
    weight_concentration_prior: float = 0.01
    max_iter: int = 200
    reg_covar: float = 1e-6
    fit_subsample: int = 100_000

    def __post_init__(self) -> None:
        super().__post_init__()
        self.feature_layers = tuple(self.feature_layers)
        validate_input_size(self.backbone, self.input_size)
        if self.covariance_type not in ("diag", "full"):
            raise ValueError(
                f"covariance_type must be 'diag' or 'full' for GPU scoring, "
                f"got {self.covariance_type!r}"
            )

    def slug_parts(self) -> list[str]:
        layers = "_".join(str(layer) for layer in self.feature_layers)
        parts = [
            short_tag(self.backbone),
            f"b{layers}",
            f"in{self.input_size}",
            f"pca{self.pca_dim}",
            f"K{self.max_components}",
            self.covariance_type,
            f"p{self.weight_concentration_prior:g}",
        ]
        if self.tta != "none":
            parts.append(f"tta-{self.tta}")
        return parts


class TorchPCA:
    """Randomised top-k PCA, fitted and applied on the GPU.

    ``svd_lowrank`` rather than a full SVD: only the top ``pca_dim``
    components are wanted and the input is typically 50k x 768, where a full
    decomposition is wasted work.
    """

    def __init__(self, dim: int, fit_subsample: int = 50_000, seed: int = 0) -> None:
        self.dim = dim
        self.fit_subsample = fit_subsample
        self.seed = seed
        self.mean: torch.Tensor | None = None
        self.components: torch.Tensor | None = None
        self.explained_variance_ratio: np.ndarray | None = None

    @torch.inference_mode()
    def fit(self, x: torch.Tensor) -> TorchPCA:
        generator = torch.Generator(device=x.device).manual_seed(self.seed)
        if x.shape[0] > self.fit_subsample:
            chosen = torch.randperm(x.shape[0], generator=generator, device=x.device)[
                : self.fit_subsample
            ]
            x = x[chosen]

        self.mean = x.mean(dim=0)
        centred = x - self.mean
        _u, singular, v = torch.svd_lowrank(
            centred, q=min(self.dim + 8, centred.shape[1]), niter=6
        )
        self.components = v[:, : self.dim].T.contiguous()

        variance = (singular**2) / max(centred.shape[0] - 1, 1)
        total = float((centred**2).sum().item()) / max(centred.shape[0] - 1, 1)
        if total > 0:
            self.explained_variance_ratio = (
                variance[: self.dim].cpu().numpy() / total
            )
        return self

    @torch.inference_mode()
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.components is None:
            raise RuntimeError("TorchPCA.transform() called before fit()")
        return (x.to(self.components.device) - self.mean) @ self.components.T


class DirichletProcessMixture:
    """scikit-learn's variational DP-GMM, with log-likelihood moved to torch.

    Fitting happens once per class on the CPU, where scikit-learn's EM is
    fine. Scoring touches every pixel of every image, so the fitted
    parameters are converted to tensors and the density evaluated in batch on
    the GPU.
    """

    def __init__(
        self,
        *,
        max_components: int = 30,
        covariance_type: str = "diag",
        weight_concentration_prior: float = 0.01,
        max_iter: int = 200,
        reg_covar: float = 1e-6,
        seed: int = 0,
    ) -> None:
        from sklearn.mixture import BayesianGaussianMixture

        self.covariance_type = covariance_type
        self.model = BayesianGaussianMixture(
            n_components=max_components,
            covariance_type=covariance_type,
            weight_concentration_prior_type="dirichlet_process",
            # Below 1 favours sparser solutions; sklearn's default of 1.0
            # tends to keep every component alive.
            weight_concentration_prior=weight_concentration_prior,
            max_iter=max_iter,
            reg_covar=reg_covar,
            init_params="kmeans",
            random_state=seed,
        )
        self._log_weights: torch.Tensor | None = None
        self._means: torch.Tensor | None = None
        self._precisions_chol: torch.Tensor | None = None
        self._log_det: torch.Tensor | None = None
        self._dim: int | None = None

    def fit(self, x: np.ndarray) -> DirichletProcessMixture:
        with warnings.catch_warnings():
            # Convergence warnings are expected and uninformative here: the
            # variational bound plateaus long before max_iter on this data.
            warnings.simplefilter("ignore")
            self.model.fit(x)
        return self

    def n_effective_components(self, threshold: float = 1e-3) -> int:
        return int((self.model.weights_ > threshold).sum())

    def to(self, device: torch.device) -> DirichletProcessMixture:
        """Cache the fitted parameters as tensors for batched scoring."""
        self._dim = self.model.means_.shape[1]
        keep = self.model.weights_ > 1e-4  # dead components cost time, not accuracy
        weights = self.model.weights_[keep]
        chol = self.model.precisions_cholesky_[keep]

        if self.covariance_type == "diag":
            log_det = np.log(chol).sum(axis=1)
        else:
            diagonal = np.arange(self._dim)
            log_det = np.log(chol[:, diagonal, diagonal]).sum(axis=1)

        self._means = torch.from_numpy(self.model.means_[keep]).to(device).float()
        self._precisions_chol = torch.from_numpy(chol).to(device).float()
        self._log_weights = (
            torch.from_numpy(np.log(np.maximum(weights, 1e-12))).to(device).float()
        )
        self._log_det = torch.from_numpy(log_det).to(device).float()
        return self

    @torch.inference_mode()
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, D)`` → ``(N,)`` log density under the mixture."""
        if self._means is None:
            raise RuntimeError("DirichletProcessMixture.log_prob() before to()")
        assert self._precisions_chol is not None
        assert self._log_weights is not None and self._log_det is not None

        const = -0.5 * self._dim * math.log(2.0 * math.pi)
        centred = x.unsqueeze(1) - self._means.unsqueeze(0)
        if self.covariance_type == "diag":
            scaled = centred * self._precisions_chol.unsqueeze(0)
        else:
            scaled = torch.einsum("kij,nkj->nki", self._precisions_chol, centred)
        mahalanobis = (scaled**2).sum(dim=-1)

        log_components = const + self._log_det.unsqueeze(0) - 0.5 * mahalanobis
        return torch.logsumexp(log_components + self._log_weights.unsqueeze(0), dim=1)


class DinoDPMM(Detector[DinoDPMMConfig]):
    name: ClassVar[str] = "dino_dpmm"
    config_type: ClassVar[type[DetectorConfig]] = DinoDPMMConfig

    def __init__(self, config: DinoDPMMConfig, runtime: RuntimeConfig) -> None:
        super().__init__(config, runtime)
        self.backbone = build_backbone(config.backbone, device=self.device)
        self.pca: TorchPCA | None = None
        self.mixture: DirichletProcessMixture | None = None

    def _patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """``(B, P, C)`` patch tokens, concatenated over requested layers."""
        maps = self.backbone(images, layers=self.config.feature_layers)
        parts = []
        for layer in sorted(maps):
            feature = maps[layer]
            batch, channels, height, width = feature.shape
            parts.append(
                feature.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
            )
        return torch.cat(parts, dim=-1).float()

    def _grid(self) -> tuple[int, int]:
        patch = patch_size_of(self.config.backbone)
        assert patch is not None
        side = self.config.input_size // patch
        return side, side

    # ── fit ──────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def fit(self, train_good: Sequence[ImageRecord]) -> None:
        self.log.info(
            "    [%s] extracting patch tokens (%d images)", now_hms(), len(train_good)
        )
        dim = self._feature_dim()
        chunks = [
            self._patch_tokens(images.to(self.device, non_blocking=True))
            .reshape(-1, dim)
            .cpu()
            for images, _masks, _indices in self.make_loader(
                train_good, batch_size=self.runtime.batch_size
            )
        ]
        features = torch.cat(chunks, dim=0)
        self.log.info("    -> %d patches, %d dims", features.shape[0], features.shape[1])

        self.pca = TorchPCA(
            self.config.pca_dim, self.config.pca_fit_subsample, self.config.seed
        ).fit(features.to(self.device))
        if self.pca.explained_variance_ratio is not None:
            self.log.info(
                "    PCA %d dims explain %.1f%% of variance",
                self.config.pca_dim,
                100 * float(self.pca.explained_variance_ratio.sum()),
            )

        projected = self.pca.transform(features.to(self.device)).cpu().numpy()
        if projected.shape[0] > self.config.fit_subsample:
            rng = np.random.default_rng(self.config.seed)
            chosen = rng.choice(
                projected.shape[0], size=self.config.fit_subsample, replace=False
            )
            projected = projected[chosen]

        self.log.info(
            "    [%s] fitting DP-GMM (max K=%d, %s) on %d patches",
            now_hms(),
            self.config.max_components,
            self.config.covariance_type,
            projected.shape[0],
        )
        self.mixture = DirichletProcessMixture(
            max_components=self.config.max_components,
            covariance_type=self.config.covariance_type,
            weight_concentration_prior=self.config.weight_concentration_prior,
            max_iter=self.config.max_iter,
            reg_covar=self.config.reg_covar,
            seed=self.config.seed,
        )
        self.mixture.fit(projected).to(self.device)
        self.log.info(
            "    [%s] %d of %d components survived pruning",
            now_hms(),
            self.mixture.n_effective_components(),
            self.config.max_components,
        )

    def _feature_dim(self) -> int:
        """Total token width after concatenating the requested layers."""
        return self.backbone.total_channels(self.config.feature_layers)

    # ── score ────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def score_batch(self, images: torch.Tensor) -> torch.Tensor:
        if self.pca is None or self.mixture is None:
            raise RuntimeError("DinoDPMM.score_batch() called before fit()")

        tokens = self._patch_tokens(images)
        batch, n_patches, channels = tokens.shape
        projected = self.pca.transform(tokens.reshape(-1, channels))
        # Negative log-likelihood: high where the patch is unlikely.
        scores = -self.mixture.log_prob(projected)
        height, width = self._grid()
        return self.upsample(scores.reshape(batch, height, width))

    def release(self) -> None:
        self.pca = None
        self.mixture = None
        super().release()
