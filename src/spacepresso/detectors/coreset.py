"""Greedy k-center coreset selection, and nearest-neighbour memory banks.

Shared by PatchCore and CutPaste — both score a patch by its distance to the
nearest patch in a bank of normal patches, and both need that bank subsampled
to a manageable size first.

Two selection algorithms:

* **exact** — the textbook greedy k-center: pick the point furthest from
  everything chosen so far, repeat. One GPU sync per selected point, so
  selecting 100k points costs 100k round trips.
* **minibatch** — pick the top-``b`` furthest points at once. Slightly worse
  coverage per point, roughly ``b`` times fewer syncs. This is the default,
  and what every run in ``baseline_out/`` used.

Both operate on a random projection of the features (Johnson–Lindenstrauss):
32 dimensions preserve pairwise distances well enough for the *ordering* that
k-center depends on, at a fraction of the memory of the full 1024-d vectors.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from spacepresso.core.logging import get_logger

__all__ = ["MemoryBank", "greedy_coreset"]

logger = get_logger(__name__)


@torch.inference_mode()
def _project(
    features: torch.Tensor,
    device: torch.device,
    seed: int,
    dim: int,
    chunk: int,
) -> torch.Tensor:
    """Random-project CPU features to ``dim`` dimensions on the GPU."""
    n, d = features.shape
    if dim >= d:
        raise ValueError(f"projection dim {dim} must be smaller than {d}")

    generator = torch.Generator(device=device).manual_seed(seed)
    projection = torch.randn(
        d, dim, generator=generator, device=device, dtype=torch.float32
    ) / math.sqrt(dim)

    out = torch.empty((n, dim), device=device, dtype=torch.float32)
    started = time.time()
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        out[start:end] = features[start:end].to(
            device, dtype=torch.float32, non_blocking=True
        ) @ projection

    del projection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(
        "      projected %d features to %d-d in %.1fs (%.1f MB)",
        n,
        dim,
        time.time() - started,
        out.element_size() * out.numel() / 1e6,
    )
    return out


@torch.inference_mode()
def _kcenter(
    points: torch.Tensor, n_select: int, seed: int, batch_size: int
) -> torch.Tensor:
    """Greedy k-center. ``batch_size=1`` is the exact algorithm."""
    device = points.device
    n = points.shape[0]
    n_select = min(n_select, n)
    if n_select <= 1:
        return torch.zeros(max(n_select, 0), dtype=torch.long, device=device)

    generator = torch.Generator(device=device).manual_seed(seed + 1)
    first = int(torch.randint(0, n, (1,), generator=generator, device=device).item())

    selected = torch.empty(n_select, dtype=torch.long, device=device)
    selected[0] = first
    min_dist = torch.cdist(points, points[first : first + 1]).squeeze(1)
    min_dist[first] = -1.0

    chosen = 1
    started = time.time()
    total_rounds = max(1, (n_select - 1 + batch_size - 1) // batch_size)
    log_every = max(1, total_rounds // 20)
    round_index = 0

    while chosen < n_select:
        take = min(batch_size, n_select - chosen)
        _, indices = torch.topk(min_dist, take, largest=True)
        selected[chosen : chosen + take] = indices
        min_dist = torch.minimum(
            min_dist, torch.cdist(points, points[indices]).min(dim=1).values
        )
        min_dist[indices] = -1.0
        chosen += take
        round_index += 1
        if round_index % log_every == 0 or chosen == n_select:
            logger.info(
                "      coreset: %d/%d (%.1f%%)  max-min-dist=%.4f  elapsed=%.1fs",
                chosen,
                n_select,
                chosen / n_select * 100,
                float(min_dist.max().clamp_min(0.0)),
                time.time() - started,
            )

    return selected


@torch.inference_mode()
def greedy_coreset(
    features: torch.Tensor,
    n_select: int,
    device: torch.device,
    *,
    seed: int = 0,
    projection_dim: int = 32,
    project_chunk: int = 65_536,
    algorithm: str = "minibatch",
    batch_size: int = 64,
) -> torch.Tensor:
    """Select ``n_select`` representative rows from CPU ``features``.

    Returns CPU indices into ``features``.
    """
    if features.device.type != "cpu":
        raise ValueError(f"expected CPU features, got {features.device}")
    if algorithm not in ("exact", "minibatch"):
        raise ValueError(
            f"coreset algorithm must be 'exact' or 'minibatch', got {algorithm!r}"
        )

    projected = _project(features, device, seed, projection_dim, project_chunk)
    selected = _kcenter(
        projected, n_select, seed, batch_size=1 if algorithm == "exact" else batch_size
    )
    out = selected.cpu()

    del projected, selected
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


class MemoryBank:
    """L2-normalised patch bank with chunked nearest-neighbour search.

    Distance is ``1 - cosine similarity``; because every vector is unit-norm
    that is a monotone function of Euclidean distance, so the *ranking* — all
    the AP metric sees — is identical while the arithmetic stays in fp16
    safely.

    Both loops are chunked. The query chunk bounds the ``(q, m)`` similarity
    matrix; the memory chunk bounds how much of the bank is resident at once.
    A 300k-patch bank against a 4096-patch query in fp32 would be a 5 GB
    intermediate, which is what the chunk sizes exist to prevent.
    """

    def __init__(self, vectors: torch.Tensor, dtype: torch.dtype = torch.float16):
        self.vectors = F.normalize(vectors.float(), p=2, dim=-1).to(dtype).contiguous()

    @property
    def size(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def memory_mb(self) -> float:
        return self.vectors.element_size() * self.vectors.numel() / 1e6

    @torch.inference_mode()
    def distance(
        self,
        queries: torch.Tensor,
        *,
        query_chunk: int = 4096,
        memory_chunk: int = 32_768,
        k: int = 1,
    ) -> torch.Tensor:
        """Mean distance to the ``k`` nearest bank entries, per query row.

        ``k > 1`` averages the top-k, which smooths the score at the cost of
        blurring small defects.
        """
        n_queries = queries.shape[0]
        out = torch.empty(n_queries, device=self.vectors.device, dtype=torch.float32)
        dtype = self.vectors.dtype

        for start in range(0, n_queries, query_chunk):
            end = min(n_queries, start + query_chunk)
            block = queries[start:end].to(self.vectors.device, dtype)

            if k == 1:
                best = torch.full(
                    (block.shape[0],), -2.0, device=block.device, dtype=dtype
                )
                for m_start in range(0, self.size, memory_chunk):
                    m_end = min(self.size, m_start + memory_chunk)
                    similarity = block @ self.vectors[m_start:m_end].T
                    torch.maximum(best, similarity.max(dim=1).values, out=best)
                out[start:end] = 1.0 - best.float()
            else:
                top = torch.full(
                    (block.shape[0], k), -2.0, device=block.device, dtype=dtype
                )
                for m_start in range(0, self.size, memory_chunk):
                    m_end = min(self.size, m_start + memory_chunk)
                    similarity = block @ self.vectors[m_start:m_end].T
                    width = min(k, similarity.shape[1])
                    merged = torch.cat(
                        [top, similarity.topk(width, dim=1).values], dim=1
                    )
                    top = merged.topk(k, dim=1).values
                out[start:end] = (1.0 - top.float()).mean(dim=1)

        return out
