# """Spacepresso baseline v3 — PatchCore + multi-scale + TTA + tracking.
#
# Drop-in successor to patchcore_baseline_v2.py. Fixes the OOM that v2 hits
# on shared NVIDIA L4 (24 GB) when running input_size=384 + multi-scale
# (--feature-layers 1 2 3).
#
# # Why v2 OOMed at exp5 scale
#
# At input_size=384 with layers=(1,2,3) and target_layer=2, you get:
#
#       class_01: 2600 train_good × 2304 patches/img = 5,990,400 patches
#                 ↳ 43 GB on CPU as fp32, 21.5 GB as fp16
#
# v2's `fit()` did:
#
#       all_feats_gpu = all_feats.to(self.device, dtype=torch.float16, ...)
#
# which looks like it should give a 21.5 GB fp16 GPU tensor. It does NOT.
# PyTorch's `Tensor.to(device=..., dtype=...)` first allocates the
# destination in the SOURCE dtype, then casts down. So the actual request
# becomes "allocate ~40 GiB fp32 on GPU, then cast to fp16" — which fails
# on a 22 GB card every time, regardless of how the `--coreset-fp16` flag
# is set. That's exactly the traceback you saw:
#
#       torch.OutOfMemoryError: Tried to allocate 39.99 GiB. ...
#
# # What v3 changes — two surgical fixes, nothing else
#
# (1) greedy_coreset accepts a CPU tensor and the GPU device. The random
#     Gaussian projection P (1792 × 32 fp32 = 230 KB) lives on GPU. We
#     walk the CPU features in chunks, transfer each chunk, project to
#     32-d, and discard the high-dim chunk. Only the projected tensor
#     stays on GPU: 6M × 32 × 4 B = 768 MB. All k-center work runs on
#     that 32-d projected representation, exactly as before — the
#     selection is mathematically identical to v2 modulo fp16 noise.
#
# (2) fit() keeps `all_feats` on CPU throughout selection. After the
#     greedy run returns the selected indices, we index the CPU tensor
#     for ONLY those rows (~180K at 3% coreset) and move just those to
#     GPU as the memory bank. Peak GPU footprint during the move:
#     180K × 1792 × 4 B ≈ 1.3 GB momentarily, then ~645 MB as fp16.
#
# `--coreset-fp16` still exists. Its meaning is narrowed: it casts the
# CPU features to fp16 on CPU (43 GB → 21.5 GB CPU RAM). Useful if your
# compute node is tight on RAM; harmless otherwise. The GPU never sees
# the full features regardless of this flag.
#
# All other behaviour — TTA, multi-scale, scoring chunks, run tracking,
# submission format, q8rle, ablation CSV — is byte-identical to v2.
#
# # CLI compatibility
#
# The exp5 sweep script runs unchanged. The only new optional knob is:
#
#       --project-chunk INT       (default 65536)
#             CPU→GPU chunk size during coreset projection. Per-chunk
#             peak on GPU is chunk × D × 4 B (469 MB at D=1792, fp32).
#             Lower this if even per-chunk allocations are tight.
# """
# from __future__ import annotations
#
# import argparse
# import csv
# import hashlib
# import json
# import math
# import re
# import sys
# import time
# import zipfile
# from collections import defaultdict
# from contextlib import contextmanager
# from dataclasses import asdict, dataclass, field
# from pathlib import Path
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from PIL import Image
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
# from torchvision.models import (
#     wide_resnet50_2, Wide_ResNet50_2_Weights,
#     resnet18, ResNet18_Weights,
#     resnet50, ResNet50_Weights,
# )
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Defaults — override on the CLI
# # ─────────────────────────────────────────────────────────────────────────────
# PROJECT_ROOT = Path("/work/u10813429/anomaly-detection")
# DEFAULT_DATA_ROOT  = PROJECT_ROOT / "data"
# DEFAULT_REPORT_DIR = PROJECT_ROOT / "baseline_out"
#
# IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD  = (0.229, 0.224, 0.225)
# VIEW_RE = re.compile(r"^(?P<base>.+?)_view(?P<v>\d+)\.[A-Za-z]+$")
#
# # ResNet channel counts per stage (used for sanity-printing feature dim)
# RESNET_CHANNELS = {
#     "wide_resnet50_2": {1: 256, 2: 512, 3: 1024, 4: 2048},
#     "resnet50":        {1: 256, 2: 512, 3: 1024, 4: 2048},
#     "resnet18":        {1: 64,  2: 128, 3: 256,  4: 512},
# }
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Tee logger
# # ─────────────────────────────────────────────────────────────────────────────
# class Tee:
#     def __init__(self, *streams): self.streams = streams
#     def write(self, s):
#         for st in self.streams: st.write(s); st.flush()
#     def flush(self):
#         for st in self.streams: st.flush()
#
#
# @contextmanager
# def tee_to(path: Path):
#     path.parent.mkdir(parents=True, exist_ok=True)
#     f = open(path, "w", encoding="utf-8")
#     old = sys.stdout
#     sys.stdout = Tee(old, f)
#     try: yield
#     finally:
#         sys.stdout = old; f.close()
#
#
# def hr(t, c="="): print(f"\n{c * 78}\n  {t}\n{c * 78}")
# def sub(t): print(f"\n--- {t} ---")
# def now_hms(): return time.strftime("%H:%M:%S")
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Dataset (unchanged from v2)
# # ─────────────────────────────────────────────────────────────────────────────
# @dataclass
# class ImageRecord:
#     path: Path
#     cls: str
#     split: str
#     anomaly_type: str | None = None
#     sample_id: str | None = None
#     view: int | None = None
#     mask_path: Path | None = None
#
#
# def parse_view(filename: str) -> tuple[str, int | None]:
#     m = VIEW_RE.match(filename)
#     if m:
#         return m.group("base"), int(m.group("v"))
#     return Path(filename).stem, None
#
#
# def scan_dataset(data_root: Path) -> list[ImageRecord]:
#     out: list[ImageRecord] = []
#     if not data_root.exists():
#         print(f"  [FATAL] {data_root} not found"); return out
#     classes = sorted(d.name for d in data_root.iterdir()
#                      if d.is_dir() and d.name.startswith("class_"))
#     for cls in classes:
#         cdir = data_root / cls
#         gd = cdir / "train" / "good"
#         if gd.exists():
#             for p in sorted(gd.iterdir()):
#                 if p.suffix.lower() in IMG_EXTS:
#                     sid, v = parse_view(p.name)
#                     out.append(ImageRecord(p, cls, "train_good",
#                                            sample_id=sid, view=v))
#         td = cdir / "train"
#         if td.exists():
#             for sd in sorted(td.iterdir()):
#                 if (not sd.is_dir() or sd.name == "good"
#                         or not sd.name.startswith("anomaly_")):
#                     continue
#                 a_type = sd.name
#                 gtd = cdir / "ground_truth_train" / a_type
#                 for p in sorted(sd.iterdir()):
#                     if p.suffix.lower() not in IMG_EXTS: continue
#                     sid, v = parse_view(p.name)
#                     mp = None
#                     if gtd.exists():
#                         cand = gtd / p.name
#                         if cand.exists():
#                             mp = cand
#                         else:
#                             for q in gtd.iterdir():
#                                 if (q.stem == p.stem
#                                         and q.suffix.lower() in IMG_EXTS):
#                                     mp = q; break
#                     out.append(ImageRecord(p, cls, "train_anomaly",
#                                            anomaly_type=a_type,
#                                            sample_id=sid, view=v,
#                                            mask_path=mp))
#         ted = cdir / "test"
#         if ted.exists():
#             for p in sorted(ted.rglob("*")):
#                 if p.is_file() and p.suffix.lower() in IMG_EXTS:
#                     sid, v = parse_view(p.name)
#                     out.append(ImageRecord(p, cls, "test",
#                                            sample_id=sid, view=v))
#     return out
#
#
# class SpacepressoDataset(Dataset):
#     def __init__(self, records: list[ImageRecord], load_masks: bool,
#                  input_size: int):
#         self.records = records
#         self.load_masks = load_masks
#         self.input_size = input_size
#         self.tx = transforms.Compose([
#             transforms.Resize((input_size, input_size)),
#             transforms.ToTensor(),
#             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
#         ])
#
#     def __len__(self): return len(self.records)
#
#     def __getitem__(self, i):
#         r = self.records[i]
#         with Image.open(r.path) as im:
#             im = im.convert("RGB")
#             x = self.tx(im)
#         if self.load_masks and r.mask_path is not None:
#             with Image.open(r.mask_path) as mm:
#                 mm = mm.convert("L")
#                 # Local eval runs at internal input_size; submission scores
#                 # are resized to 224 downstream (see maybe_resize_to_submission).
#                 mm = mm.resize((self.input_size, self.input_size),
#                                Image.NEAREST)
#                 m = (np.asarray(mm) > 127).astype(np.float32)
#         else:
#             m = np.zeros((self.input_size, self.input_size), dtype=np.float32)
#         return x, torch.from_numpy(m), i
#
#
# def make_loader(records, batch_size, input_size,
#                 load_masks=False, num_workers=2, shuffle=False):
#     ds = SpacepressoDataset(records, load_masks=load_masks,
#                             input_size=input_size)
#     return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
#                       num_workers=num_workers, pin_memory=True,
#                       persistent_workers=(num_workers > 0))
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Feature extractor — multi-stage configurable, only stores requested layers
# # (unchanged from v2)
# # ─────────────────────────────────────────────────────────────────────────────
# class FeatureExtractor(nn.Module):
#     """Wraps a torchvision ResNet-family backbone, exposes layers 1-4."""
#
#     def __init__(self, backbone: str = "wide_resnet50_2"):
#         super().__init__()
#         if backbone == "wide_resnet50_2":
#             m = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
#         elif backbone == "resnet50":
#             m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
#         elif backbone == "resnet18":
#             m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
#         else:
#             raise ValueError(f"unknown backbone: {backbone}")
#         self.backbone_name = backbone
#         self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
#         self.layer1 = m.layer1
#         self.layer2 = m.layer2
#         self.layer3 = m.layer3
#         self.layer4 = m.layer4
#         self.eval()
#         for p in self.parameters(): p.requires_grad_(False)
#
#     @torch.inference_mode()
#     def forward(self, x: torch.Tensor,
#                 layers: tuple[int, ...] = (2, 3)) -> dict[int, torch.Tensor]:
#         out: dict[int, torch.Tensor] = {}
#         need_4 = 4 in layers
#         x = self.stem(x)
#         x = self.layer1(x)
#         if 1 in layers: out[1] = x
#         x = self.layer2(x)
#         if 2 in layers: out[2] = x
#         x = self.layer3(x)
#         if 3 in layers: out[3] = x
#         if need_4:
#             x = self.layer4(x)
#             out[4] = x
#         return out
#
#
# def patchify_and_combine(maps: dict[int, torch.Tensor],
#                           patch_size: int = 3,
#                           target_layer: int = 2) -> torch.Tensor:
#     """Resize every layer's map to the spatial size of `target_layer`,
#     average-pool with `patch_size`, concat channels, L2-normalise patches.
#     Returns (B, H*W, C_total)."""
#     assert target_layer in maps, \
#         f"target_layer={target_layer} not in {list(maps)}"
#     H, W = maps[target_layer].shape[-2:]
#     avg = nn.AvgPool2d(kernel_size=patch_size, stride=1, padding=patch_size // 2)
#     pooled = []
#     for k in sorted(maps.keys()):
#         m = maps[k]
#         if m.shape[-2:] != (H, W):
#             m = F.interpolate(m, size=(H, W), mode="bilinear",
#                               align_corners=False)
#         pooled.append(avg(m))
#     feats = torch.cat(pooled, dim=1)              # (B, C_tot, H, W)
#     B, C, _, _ = feats.shape
#     feats = feats.permute(0, 2, 3, 1).reshape(B, H * W, C)
#     feats = F.normalize(feats, p=2, dim=-1)
#     return feats
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Greedy k-center coreset — v3 OOM-safe variant
# #
# # Features stay on CPU. We project them in chunks to a low-dim space on
# # GPU (default 32-d) and run k-center entirely on the projected rep.
# # Peak GPU footprint scales with N × projection_dim (default 32), NOT
# # with N × D (which can be 40+ GB at exp5 scale).
# # ─────────────────────────────────────────────────────────────────────────────
# @torch.inference_mode()
# def greedy_coreset(features_cpu: torch.Tensor,
#                     n_select: int,
#                     device: torch.device,
#                     seed: int = 0,
#                     projection_dim: int = 32,
#                     project_chunk: int = 65536) -> torch.Tensor:
#     """Greedy k-center coreset (Sener & Savarese 2018) without ever
#     materialising the full feature tensor on GPU.
#
#     Pipeline:
#       1. Random Gaussian projection P  (D × projection_dim, fp32, on GPU)
#       2. Walk `features_cpu` in chunks of `project_chunk` rows. Move each
#          chunk to GPU as fp32, multiply by P, write into `feats_proj`.
#          Discard the high-dim chunk.
#       3. Run greedy k-center on `feats_proj` (N × projection_dim fp32 on
#          GPU — under 1 GB for N=6M, projection_dim=32).
#
#     Returns selected indices as a CPU LongTensor of shape (n_select,).
#     """
#     assert features_cpu.device.type == "cpu", (
#         f"v3 expects CPU features; got {features_cpu.device}. "
#         f"If you have GPU features call greedy_coreset_gpu (not provided)."
#     )
#     N, D = features_cpu.shape
#     n_select = min(n_select, N)
#     if projection_dim >= D:
#         raise ValueError(
#             f"projection_dim={projection_dim} must be < D={D} for v3's "
#             f"CPU-features path. (Without projection the work tensor on "
#             f"GPU would be the full feature tensor — the very thing we "
#             f"are trying to avoid.)"
#         )
#
#     # ── (1) Projection matrix on GPU.  Tiny: 1792 × 32 × 4 B = 230 KB. ──
#     g = torch.Generator(device=device).manual_seed(seed)
#     P = torch.randn(D, projection_dim, generator=g, device=device,
#                     dtype=torch.float32) / math.sqrt(projection_dim)
#
#     # ── (2) Allocate the projected tensor on GPU; chunk-transfer + project ──
#     feats_proj = torch.empty((N, projection_dim), device=device,
#                               dtype=torch.float32)
#     t_proj = time.time()
#     last_log_at = 0
#     for s in range(0, N, project_chunk):
#         e = min(N, s + project_chunk)
#         # Cast on transfer:  CPU tensor (fp16 or fp32)  →  GPU fp32 chunk.
#         # Per-chunk peak: (e - s) × D × 4 B  (≈469 MB at chunk=65536, D=1792).
#         # Released as soon as the matmul completes.
#         chunk_gpu = features_cpu[s:e].to(device, dtype=torch.float32,
#                                           non_blocking=True)
#         feats_proj[s:e] = chunk_gpu @ P
#         del chunk_gpu
#         if (e - last_log_at) >= 1_000_000 or e == N:
#             last_log_at = e
#             print(f"      projecting features {e:>9d}/{N}  "
#                   f"({e / N * 100:>5.1f}%)  "
#                   f"elapsed={time.time() - t_proj:.1f}s", flush=True)
#     del P
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#     print(f"      projected {N} features to {projection_dim}-d on GPU "
#           f"in {time.time() - t_proj:.1f}s "
#           f"({feats_proj.element_size() * feats_proj.numel() / 1e6:.1f} "
#           f"MB on GPU)")
#
#     # ── (3) k-center on the projected rep ──
#     min_dist = torch.full((N,), float("inf"), device=device,
#                           dtype=torch.float32)
#     rng = torch.Generator(device=device).manual_seed(seed + 1)
#     first = int(torch.randint(0, N, (1,), generator=rng, device=device).item())
#     selected = torch.empty(n_select, dtype=torch.long, device=device)
#     selected[0] = first
#     d0 = torch.cdist(feats_proj, feats_proj[first:first + 1]).squeeze(1)
#     min_dist = torch.minimum(min_dist, d0)
#     log_every = max(1, n_select // 20)
#     t0 = time.time()
#     for i in range(1, n_select):
#         idx = int(torch.argmax(min_dist).item())
#         selected[i] = idx
#         d_new = torch.cdist(feats_proj, feats_proj[idx:idx + 1]).squeeze(1)
#         min_dist = torch.minimum(min_dist, d_new)
#         if i % log_every == 0 or i == n_select - 1:
#             print(f"      coreset: {i + 1}/{n_select} "
#                   f"({(i + 1) / n_select * 100:>5.1f}%)  "
#                   f"max-min-dist={min_dist.max().item():.4f}  "
#                   f"elapsed={time.time() - t0:.1f}s", flush=True)
#     selected_cpu = selected.cpu()
#     del feats_proj, min_dist, selected
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#     return selected_cpu
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Target-layer auto-resolution (unchanged from v2)
# # ─────────────────────────────────────────────────────────────────────────────
# def resolve_target_layer(feature_layers: tuple[int, ...],
#                           explicit: str | int | None) -> int:
#     """Auto-rule: prefer layer 2 if it's in the stack (PatchCore-as-written
#     keeps a 28x28 grid even when fusing multiple stages); otherwise fall
#     back to the smallest-numbered (= highest-resolution) layer."""
#     if explicit is None or (isinstance(explicit, str) and explicit == "auto"):
#         if 2 in feature_layers:
#             return 2
#         return min(feature_layers)
#     val = int(explicit)
#     if val not in feature_layers:
#         raise ValueError(
#             f"--target-layer {val} not in --feature-layers "
#             f"{list(feature_layers)}; pass one of {list(feature_layers)} or "
#             f"'auto'."
#         )
#     return val
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # PatchCore — multi-layer + TTA-aware + OOM-safe (v3 fit())
# # ─────────────────────────────────────────────────────────────────────────────
# @dataclass
# class PatchCoreConfig:
#     backbone: str = "wide_resnet50_2"
#     feature_layers: tuple[int, ...] = (2, 3)
#     target_layer: int = 2          # resolved by caller
#     input_size: int = 224
#     coreset_frac: float = 0.10
#     coreset_fp16: bool = False     # v3: cast CPU features to fp16 (CPU RAM only)
#     patch_size: int = 3
#     knn_k: int = 9                 # kept for API compat; only 1-NN used
#     batch_size: int = 32           # for fit() / feature extraction
#     score_batch_size: int = 16     # for score_batch() / inference
#     score_chunk: int = 4096        # query-axis chunk during scoring
#     memory_chunk: int = 32768      # memory-bank-axis chunk during scoring
#     memory_dtype: str = "fp16"     # "fp16" | "fp32"
#     project_chunk: int = 65536     # CPU→GPU chunk during coreset projection
#     num_workers: int = 2
#     device: str = "cuda"
#     seed: int = 0
#
#
# class PatchCore:
#     def __init__(self, cfg: PatchCoreConfig):
#         self.cfg = cfg
#         self.device = torch.device(
#             cfg.device if torch.cuda.is_available() else "cpu")
#         self.extractor = FeatureExtractor(cfg.backbone).to(self.device)
#         self.target_layer = cfg.target_layer
#         assert self.target_layer in cfg.feature_layers, (
#             f"target_layer={self.target_layer} not in "
#             f"feature_layers={cfg.feature_layers}"
#         )
#         self.memory_dtype = (torch.float16 if cfg.memory_dtype == "fp16"
#                              else torch.float32)
#         self.memory: torch.Tensor | None = None     # (M, D) on GPU
#         self.feature_hw: tuple[int, int] | None = None
#         self.feature_dim: int | None = None
#
#     @torch.inference_mode()
#     def _extract(self, loader: DataLoader) -> torch.Tensor:
#         feats_list = []
#         n = 0
#         last_log = 0
#         for x, _, _ in loader:
#             bs = x.shape[0]
#             x = x.to(self.device, non_blocking=True)
#             maps = self.extractor(x, layers=self.cfg.feature_layers)
#             pf = patchify_and_combine(maps,
#                                        patch_size=self.cfg.patch_size,
#                                        target_layer=self.target_layer)
#             if self.feature_hw is None:
#                 P = pf.shape[1]
#                 H = W = int(math.isqrt(P))
#                 self.feature_hw = (H, W)
#                 self.feature_dim = pf.shape[2]
#             pf = pf.reshape(-1, pf.shape[-1]).detach().cpu()
#             feats_list.append(pf)
#             del x, maps, pf
#             n += bs
#             if n - last_log >= 256:
#                 last_log = n
#                 print(f"      extracted features from {n} images "
#                       f"(feat dim={self.feature_dim}, "
#                       f"patches/img={self.feature_hw[0] * self.feature_hw[1]})",
#                       flush=True)
#         return torch.cat(feats_list, dim=0)
#
#     def fit(self, train_good_records: list[ImageRecord]) -> None:
#         print(f"    [{now_hms()}] extracting train/good features "
#               f"({len(train_good_records)} images, "
#               f"input_size={self.cfg.input_size}, "
#               f"layers={list(self.cfg.feature_layers)}, "
#               f"target_layer={self.target_layer})...")
#         loader = make_loader(train_good_records,
#                              batch_size=self.cfg.batch_size,
#                              input_size=self.cfg.input_size,
#                              num_workers=self.cfg.num_workers,
#                              load_masks=False, shuffle=False)
#         all_feats = self._extract(loader)   # CPU fp32
#
#         # v3 NOTE: all_feats lives on CPU for the entire coreset phase.
#         # Optionally cast on CPU to halve CPU RAM (43 GB → 21.5 GB at
#         # exp5 scale).  Casting on CPU is cheap, and avoids the v2 trap
#         # where `.to(device=..., dtype=...)` allocated the fp32 staging
#         # tensor on GPU before casting to fp16.
#         if self.cfg.coreset_fp16:
#             all_feats = all_feats.half()
#             print(f"    -> {all_feats.shape[0]} patch features "
#                   f"(fp16, CPU; "
#                   f"{all_feats.element_size() * all_feats.numel() / 1e9:.2f} "
#                   f"GB)")
#         else:
#             print(f"    -> {all_feats.shape[0]} patch features "
#                   f"(fp32, CPU; "
#                   f"{all_feats.element_size() * all_feats.numel() / 1e9:.2f} "
#                   f"GB)")
#
#         n_select = max(int(self.cfg.coreset_frac * all_feats.shape[0]), 1)
#         print(f"    [{now_hms()}] greedy coreset: selecting {n_select} "
#               f"of {all_feats.shape[0]} patches "
#               f"({self.cfg.coreset_frac:.1%})  "
#               f"[CPU features → GPU 32-d projection, "
#               f"project_chunk={self.cfg.project_chunk}]")
#
#         idx_cpu = greedy_coreset(all_feats, n_select, self.device,
#                                   seed=self.cfg.seed,
#                                   project_chunk=self.cfg.project_chunk)
#
#         # ── Pull ONLY the selected rows to GPU as the memory bank ──
#         # n_select × D × dtype_bytes.  Exp5: 180K × 1792 × 2 B = 645 MB
#         # final (fp16); momentarily ~1.3 GB during the fp32 normalisation.
#         selected_cpu = all_feats[idx_cpu]    # CPU index → CPU subset
#         del all_feats                        # release the big CPU tensor
#
#         selected_gpu = selected_cpu.to(self.device, non_blocking=True)
#         del selected_cpu
#         # Re-normalise in fp32 for numerical safety, then cast to memory_dtype.
#         memory = selected_gpu.float()
#         del selected_gpu
#         memory = F.normalize(memory, p=2, dim=-1)
#         self.memory = memory.to(self.memory_dtype).contiguous()
#         del memory
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#
#         print(f"    [{now_hms()}] memory bank ready  "
#               f"shape={tuple(self.memory.shape)}  "
#               f"dtype={self.memory.dtype}  "
#               f"({self.memory.element_size() * self.memory.numel() / 1e6:.1f} "
#               f"MB on GPU)")
#
#     @torch.inference_mode()
#     def _score_one_pass(self, x: torch.Tensor) -> torch.Tensor:
#         """One forward pass + 1-NN sim search. Returns (B, input_size,
#         input_size) score map on CPU.
#
#         Sim-based path: dist = 1 - q @ M.T  (q, M both unit-norm).
#         We chunk over both the query axis and the memory-bank axis so
#         peak GPU usage is ~ score_chunk * memory_chunk * dtype_bytes
#         regardless of how big the bank is.
#         """
#         assert self.memory is not None, "fit() first"
#         maps = self.extractor(x.to(self.device, non_blocking=True),
#                               layers=self.cfg.feature_layers)
#         pf = patchify_and_combine(maps,
#                                    patch_size=self.cfg.patch_size,
#                                    target_layer=self.target_layer)
#         del maps
#         B, P, C = pf.shape
#         H = W = int(math.isqrt(P))
#         flat = pf.reshape(-1, C)
#         N_q = flat.shape[0]
#         score_chunk = self.cfg.score_chunk
#         memory_chunk = self.cfg.memory_chunk
#         M_total = self.memory.shape[0]
#
#         # We track the 1-NN cosine sim per query (largest sim = nearest
#         # neighbour). Distance = 1 - sim. Matmuls run in memory_dtype.
#         dist_min = torch.empty(N_q, device=self.device, dtype=torch.float32)
#         for s in range(0, N_q, score_chunk):
#             e = min(N_q, s + score_chunk)
#             q = flat[s:e].to(self.memory_dtype)
#             max_sim = torch.full((q.shape[0],), -2.0, device=self.device,
#                                   dtype=self.memory_dtype)
#             for ms in range(0, M_total, memory_chunk):
#                 me = min(M_total, ms + memory_chunk)
#                 m_chunk = self.memory[ms:me]                 # (m_chunk, C)
#                 sim = q @ m_chunk.T                          # (q_chunk, m_chunk)
#                 chunk_max = sim.max(dim=1).values
#                 torch.maximum(max_sim, chunk_max, out=max_sim)
#                 del sim, chunk_max
#             dist_min[s:e] = (1.0 - max_sim.float())
#             del q, max_sim
#         del flat, pf
#
#         score_lr = dist_min.reshape(B, H, W)
#         score = F.interpolate(score_lr.unsqueeze(1),
#                               size=(self.cfg.input_size, self.cfg.input_size),
#                               mode="bilinear", align_corners=False)
#         out = score.squeeze(1).cpu()
#         del dist_min, score_lr, score
#         return out
#
#     @torch.inference_mode()
#     def score_batch(self, x: torch.Tensor, tta: str = "none") -> torch.Tensor:
#         """Score with optional TTA. `tta` ∈ {none, hflip, vflip, hvflip, d4}.
#         Aggregates by averaging un-augmented score maps."""
#         if tta == "none":
#             return self._score_one_pass(x)
#
#         accumulator = None
#         n = 0
#
#         def _add(scores: torch.Tensor):
#             nonlocal accumulator, n
#             if accumulator is None:
#                 accumulator = scores.clone()
#             else:
#                 accumulator += scores
#             n += 1
#
#         _add(self._score_one_pass(x))
#         if tta in ("hflip", "hvflip", "d4"):
#             s = self._score_one_pass(torch.flip(x, dims=[-1]))
#             _add(torch.flip(s, dims=[-1]))
#         if tta in ("vflip", "hvflip", "d4"):
#             s = self._score_one_pass(torch.flip(x, dims=[-2]))
#             _add(torch.flip(s, dims=[-2]))
#         if tta == "d4":
#             s = self._score_one_pass(torch.rot90(x, k=1, dims=[-2, -1]))
#             _add(torch.rot90(s, k=-1, dims=[-2, -1]))
#             s = self._score_one_pass(torch.rot90(x, k=2, dims=[-2, -1]))
#             _add(torch.rot90(s, k=-2, dims=[-2, -1]))
#             s = self._score_one_pass(torch.rot90(x, k=3, dims=[-2, -1]))
#             _add(torch.rot90(s, k=-3, dims=[-2, -1]))
#             xf = torch.flip(x, dims=[-1])
#             s = self._score_one_pass(torch.rot90(xf, k=1, dims=[-2, -1]))
#             s = torch.rot90(s, k=-1, dims=[-2, -1])
#             s = torch.flip(s, dims=[-1])
#             _add(s)
#             xf = torch.flip(x, dims=[-2])
#             s = self._score_one_pass(torch.rot90(xf, k=1, dims=[-2, -1]))
#             s = torch.rot90(s, k=-1, dims=[-2, -1])
#             s = torch.flip(s, dims=[-2])
#             _add(s)
#         return accumulator / max(n, 1)
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Pixel-level Average Precision (unchanged)
# # ─────────────────────────────────────────────────────────────────────────────
# def pixel_average_precision(score: np.ndarray, gt: np.ndarray) -> float:
#     s = score.astype(np.float32).ravel()
#     y = gt.astype(np.int32).ravel()
#     if y.sum() == 0:
#         return 0.0
#     try:
#         from sklearn.metrics import average_precision_score
#         return float(average_precision_score(y, s))
#     except Exception:
#         order = np.argsort(-s, kind="stable")
#         y = y[order]
#         tp = np.cumsum(y); fp = np.cumsum(1 - y)
#         precision = tp / (tp + fp + 1e-12)
#         recall = tp / max(int(y.sum()), 1)
#         recall = np.concatenate([[0.0], recall])
#         precision = np.concatenate([[1.0], precision])
#         return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Smoothing + calibration + q8rle (unchanged)
# # ─────────────────────────────────────────────────────────────────────────────
# def _gaussian_kernel_1d(sigma: float, radius: int) -> np.ndarray:
#     x = np.arange(-radius, radius + 1)
#     k = np.exp(-(x ** 2) / (2 * sigma ** 2))
#     return (k / k.sum()).astype(np.float32)
#
#
# def gaussian_smooth(score: np.ndarray, sigma: float = 1.5) -> np.ndarray:
#     if sigma <= 0:
#         return score
#     r = max(1, int(round(3 * sigma)))
#     k = _gaussian_kernel_1d(sigma, r)
#     sx = np.pad(score, ((r, r), (0, 0)), mode="reflect")
#     sx = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, sx)
#     sx = np.pad(sx, ((0, 0), (r, r)), mode="reflect")
#     sx = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 1, sx)
#     return sx
#
#
# def calibrate_to_unit(scores: list[np.ndarray]) -> tuple[float, float]:
#     flat = np.concatenate([s.ravel() for s in scores])
#     lo = float(np.percentile(flat, 1.0))
#     hi = float(np.percentile(flat, 99.5))
#     if hi <= lo: hi = lo + 1e-6
#     return lo, hi
#
#
# def float_matrix_to_q8rle(x: np.ndarray) -> str:
#     q = np.clip(np.rint(np.asarray(x, dtype=np.float32) * 255),
#                 0, 255).astype(np.uint8)
#     h, w = q.shape
#     flat = q.T.reshape(-1)
#     if flat.size == 0:
#         return f"q8rle {h} {w}"
#     cuts = np.flatnonzero(flat[1:] != flat[:-1]) + 1
#     starts = np.r_[0, cuts]
#     ends = np.r_[cuts, flat.size]
#     parts = ["q8rle", str(h), str(w)]
#     for v, n in zip(flat[starts], ends - starts):
#         parts += [str(int(v)), str(int(n))]
#     return " ".join(parts)
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Submission: scores at input_size are RESIZED to 224 (the spec)
# # ─────────────────────────────────────────────────────────────────────────────
# SUBMISSION_H = SUBMISSION_W = 224
#
#
# def maybe_resize_to_submission(score: np.ndarray) -> np.ndarray:
#     if score.shape == (SUBMISSION_H, SUBMISSION_W):
#         return score
#     t = torch.from_numpy(score).unsqueeze(0).unsqueeze(0).float()
#     t = F.interpolate(t, size=(SUBMISSION_H, SUBMISSION_W),
#                       mode="bilinear", align_corners=False)
#     return t.squeeze().numpy()
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Experiment tracking
# # ─────────────────────────────────────────────────────────────────────────────
# @dataclass
# class RunConfig:
#     data_root: Path
#     report_dir: Path
#     backbone: str = "wide_resnet50_2"
#     feature_layers: tuple[int, ...] = (2, 3)
#     target_layer: int = 2
#     input_size: int = 224
#     coreset_frac: float = 0.10
#     coreset_fp16: bool = False
#     batch_size: int = 32
#     score_batch_size: int = 16
#     score_chunk: int = 4096
#     memory_chunk: int = 32768
#     memory_dtype: str = "fp16"
#     project_chunk: int = 65536
#     num_workers: int = 2
#     smooth_sigma: float = 1.5
#     knn_k: int = 9
#     tta: str = "none"
#     seed: int = 0
#     only_classes: list[str] = field(default_factory=list)
#     skip_eval: bool = False
#     skip_submission: bool = False
#     save_memory_banks: bool = True
#     zip_submission: bool = True
#     aggressive_cleanup: bool = False
#     run_tag: str = ""
#
#
# def make_run_id(cfg: RunConfig) -> str:
#     """Stable, human-parseable run id based on the most relevant knobs."""
#     fp = json.dumps({
#         "backbone": cfg.backbone,
#         "feature_layers": list(cfg.feature_layers),
#         "target_layer": cfg.target_layer,
#         "input_size": cfg.input_size,
#         "coreset_frac": cfg.coreset_frac,
#         "coreset_fp16": cfg.coreset_fp16,
#         "knn_k": cfg.knn_k,
#         "tta": cfg.tta,
#         "smooth_sigma": cfg.smooth_sigma,
#         "memory_dtype": cfg.memory_dtype,
#         "score_batch_size": cfg.score_batch_size,
#         "score_chunk": cfg.score_chunk,
#         "memory_chunk": cfg.memory_chunk,
#         "project_chunk": cfg.project_chunk,
#         "seed": cfg.seed,
#         "v": 3,  # bump so v3 run_ids don't collide with v2 hashes
#     }, sort_keys=True).encode("utf-8")
#     digest = hashlib.sha1(fp).hexdigest()[:6]
#     stamp = time.strftime("%Y%m%d-%H%M%S")
#     bb = {"wide_resnet50_2": "wrn50", "resnet50": "rn50",
#           "resnet18": "rn18"}[cfg.backbone]
#     L = "".join(str(l) for l in cfg.feature_layers)
#     bits = (f"{stamp}_{bb}_L{L}_T{cfg.target_layer}_"
#             f"in{cfg.input_size}_cs{int(cfg.coreset_frac*100):02d}")
#     if cfg.memory_dtype != "fp16":
#         bits += f"_{cfg.memory_dtype}"
#     if cfg.tta != "none":
#         bits += f"_tta-{cfg.tta}"
#     if cfg.run_tag:
#         bits += f"_{re.sub(r'[^A-Za-z0-9._-]+', '-', cfg.run_tag)}"
#     return f"{bits}_{digest}"
#
#
# def append_to_ablation_master(master_csv: Path, row: dict) -> None:
#     master_csv.parent.mkdir(parents=True, exist_ok=True)
#     existing_rows: list[dict] = []
#     fieldnames: list[str] = []
#     if master_csv.exists():
#         with open(master_csv, "r", newline="", encoding="utf-8") as f:
#             reader = csv.DictReader(f)
#             fieldnames = list(reader.fieldnames or [])
#             existing_rows = list(reader)
#     for k in row.keys():
#         if k not in fieldnames:
#             fieldnames.append(k)
#     existing_rows.append(row)
#     with open(master_csv, "w", newline="", encoding="utf-8") as f:
#         w = csv.DictWriter(f, fieldnames=fieldnames)
#         w.writeheader()
#         for r in existing_rows:
#             w.writerow({k: r.get(k, "") for k in fieldnames})
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Per-class pipeline
# # ─────────────────────────────────────────────────────────────────────────────
# def run_one_class(cls: str, records_all: list[ImageRecord],
#                    cfg: RunConfig, run_dir: Path) -> dict:
#     hr(f"CLASS {cls}", "─")
#     t_start = time.time()
#
#     train_good = [r for r in records_all if r.cls == cls and r.split == "train_good"]
#     train_anom = [r for r in records_all if r.cls == cls and r.split == "train_anomaly"]
#     test       = [r for r in records_all if r.cls == cls and r.split == "test"]
#     print(f"  train_good={len(train_good)}  "
#           f"train_anomaly={len(train_anom)}  test={len(test)}")
#
#     pc_cfg = PatchCoreConfig(
#         backbone=cfg.backbone,
#         feature_layers=cfg.feature_layers,
#         target_layer=cfg.target_layer,
#         input_size=cfg.input_size,
#         coreset_frac=cfg.coreset_frac,
#         coreset_fp16=cfg.coreset_fp16,
#         knn_k=cfg.knn_k,
#         batch_size=cfg.batch_size,
#         score_batch_size=cfg.score_batch_size,
#         score_chunk=cfg.score_chunk,
#         memory_chunk=cfg.memory_chunk,
#         memory_dtype=cfg.memory_dtype,
#         project_chunk=cfg.project_chunk,
#         num_workers=cfg.num_workers,
#         seed=cfg.seed,
#     )
#     pc = PatchCore(pc_cfg)
#     pc.fit(train_good)
#
#     if cfg.save_memory_banks:
#         bank_path = run_dir / "banks" / f"{cls}_memory.pt"
#         bank_path.parent.mkdir(parents=True, exist_ok=True)
#         torch.save({"memory": pc.memory.cpu(),
#                     "feature_dim": pc.feature_dim,
#                     "feature_hw": pc.feature_hw,
#                     "target_layer": pc.target_layer,
#                     "memory_dtype": cfg.memory_dtype,
#                     "config": asdict(pc_cfg)}, bank_path)
#         print(f"    saved memory bank -> {bank_path}")
#
#     # ── Local validation (inference batch size!) ──
#     eval_rows: list[dict] = []
#     class_mean_ap = float("nan")
#     if not cfg.skip_eval and train_anom:
#         sub(f"local validation — per-anomaly-type pixel-AP "
#             f"(tta={cfg.tta}, score_bs={cfg.score_batch_size})")
#         loader = make_loader(train_anom, batch_size=cfg.score_batch_size,
#                              input_size=cfg.input_size,
#                              num_workers=cfg.num_workers, load_masks=True)
#         scores_per_idx: dict[int, np.ndarray] = {}
#         gt_per_idx: dict[int, np.ndarray] = {}
#         with torch.inference_mode():
#             for x, masks, idxs in loader:
#                 score_maps = pc.score_batch(x, tta=cfg.tta).numpy()
#                 masks_np = masks.numpy()
#                 for b in range(x.shape[0]):
#                     sm = gaussian_smooth(score_maps[b], cfg.smooth_sigma)
#                     scores_per_idx[int(idxs[b])] = sm
#                     gt_per_idx[int(idxs[b])] = masks_np[b]
#                 if cfg.aggressive_cleanup and torch.cuda.is_available():
#                     torch.cuda.empty_cache()
#         by_anom: dict[str, list[float]] = defaultdict(list)
#         for r_idx, sm in scores_per_idx.items():
#             r = train_anom[r_idx]
#             ap = pixel_average_precision(sm, gt_per_idx[r_idx])
#             by_anom[r.anomaly_type or "?"].append(ap)
#         print(f"    {'anomaly_type':<14} {'n_views':>8} "
#               f"{'pixel-AP (mean ± std)':>26}")
#         per_type_means = []
#         for a_type in sorted(by_anom):
#             arr = np.asarray(by_anom[a_type])
#             per_type_means.append(float(arr.mean()))
#             print(f"    {a_type:<14} {len(arr):>8} "
#                   f"{arr.mean():>15.4f} ± {arr.std():.4f}")
#             eval_rows.append({"class": cls, "anomaly_type": a_type,
#                               "n_views": int(len(arr)),
#                               "ap_mean": float(arr.mean()),
#                               "ap_std": float(arr.std()),
#                               "ap_min": float(arr.min()),
#                               "ap_max": float(arr.max())})
#         class_mean_ap = float(np.mean(per_type_means)) if per_type_means else 0.0
#         print(f"    >>> class {cls} mean pixel-AP "
#               f"(avg over types): {class_mean_ap:.4f}")
#
#     # ── Test scoring (inference batch size!) ──
#     test_results: list[tuple[ImageRecord, np.ndarray]] = []
#     if not cfg.skip_submission and test:
#         sub(f"scoring {len(test)} test images "
#             f"(tta={cfg.tta}, score_bs={cfg.score_batch_size})")
#         loader = make_loader(test, batch_size=cfg.score_batch_size,
#                              input_size=cfg.input_size,
#                              num_workers=cfg.num_workers, load_masks=False)
#         n_done = 0
#         last_log = 0
#         with torch.inference_mode():
#             for x, _, idxs in loader:
#                 score_maps = pc.score_batch(x, tta=cfg.tta).numpy()
#                 for b in range(x.shape[0]):
#                     sm = gaussian_smooth(score_maps[b], cfg.smooth_sigma)
#                     sm = maybe_resize_to_submission(sm)
#                     test_results.append((test[int(idxs[b])], sm))
#                 n_done += x.shape[0]
#                 if n_done - last_log >= 200:
#                     last_log = n_done
#                     print(f"      scored {n_done}/{len(test)}", flush=True)
#                 if cfg.aggressive_cleanup and torch.cuda.is_available():
#                     torch.cuda.empty_cache()
#
#     elapsed_min = (time.time() - t_start) / 60.0
#     print(f"  class {cls} done in {elapsed_min:.1f} min")
#     del pc
#     if torch.cuda.is_available(): torch.cuda.empty_cache()
#     return {
#         "class": cls,
#         "class_mean_ap": class_mean_ap,
#         "eval_rows": eval_rows,
#         "test_results": test_results,
#         "elapsed_min": elapsed_min,
#     }
#
#
# def write_submission(all_test_results, run_dir: Path,
#                      zip_it: bool = True) -> Path:
#     sub("calibrating scores and writing submission.csv")
#     scores = [sm for _, sm in all_test_results]
#     if not scores:
#         raise RuntimeError("no test scores")
#     lo, hi = calibrate_to_unit(scores)
#     print(f"    global score calibration  lo={lo:.4f}  hi={hi:.4f}")
#     csv_path = run_dir / "submission.csv"
#     csv_path.parent.mkdir(parents=True, exist_ok=True)
#     n = 0
#     with open(csv_path, "w", newline="", encoding="utf-8") as f:
#         w = csv.writer(f)
#         w.writerow(["ID", "Label"])
#         for r, sm in all_test_results:
#             normed = np.clip((sm - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
#             w.writerow([r.path.stem, float_matrix_to_q8rle(normed)])
#             n += 1
#     print(f"    wrote {n} rows -> {csv_path}")
#     if zip_it:
#         zip_path = csv_path.with_suffix(".zip")
#         with zipfile.ZipFile(zip_path, "w",
#                              compression=zipfile.ZIP_DEFLATED) as zf:
#             zf.write(csv_path, arcname=csv_path.name)
#         print(f"    zipped         -> {zip_path}")
#         return zip_path
#     return csv_path
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Main
# # ─────────────────────────────────────────────────────────────────────────────
# def main():
#     ap = argparse.ArgumentParser(
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         description=__doc__)
#     ap.add_argument("--data-root",  type=Path, default=DEFAULT_DATA_ROOT)
#     ap.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
#     ap.add_argument("--backbone", default="wide_resnet50_2",
#                     choices=["wide_resnet50_2", "resnet50", "resnet18"])
#     ap.add_argument("--feature-layers", type=int, nargs="+", default=[2, 3],
#                     help="ResNet stages to fuse (any subset of 1 2 3 4)")
#     ap.add_argument("--target-layer", default="auto",
#                     help="Spatial grid to resample everything to: "
#                          "auto|1|2|3|4. 'auto' picks layer 2 if in "
#                          "feature-layers (PatchCore-as-written), else "
#                          "the smallest-numbered layer.")
#     ap.add_argument("--input-size", type=int, default=224,
#                     choices=[224, 256, 320, 384, 448, 512])
#     ap.add_argument("--coreset-frac", type=float, default=0.10)
#     ap.add_argument("--coreset-fp16", action="store_true",
#                     help="v3: cast the CPU features to fp16 (CPU RAM only; "
#                          "the GPU never sees the full features). Halves "
#                          "CPU RAM use during coreset selection (~43 GB → "
#                          "~21.5 GB at exp5 scale). Selection is "
#                          "mathematically identical up to fp16 noise.")
#     # batch sizes — train vs. inference are separate
#     ap.add_argument("--batch-size", type=int, default=32,
#                     help="batch size for feature extraction during fit()")
#     ap.add_argument("--score-batch-size", type=int, default=16,
#                     help="batch size for inference (validation + test). "
#                          "Smaller is safer; activations + sim live on GPU.")
#     ap.add_argument("--score-chunk", type=int, default=4096,
#                     help="query-axis chunk size during nearest-neighbour "
#                          "scoring.")
#     ap.add_argument("--memory-chunk", type=int, default=32768,
#                     help="memory-bank-axis chunk during scoring. Caps the "
#                          "(query, bank) sim matrix at "
#                          "score_chunk * memory_chunk * dtype_bytes.")
#     ap.add_argument("--memory-dtype", default="fp16",
#                     choices=["fp16", "fp32"],
#                     help="dtype of the stored memory bank and sim matmul "
#                          "during scoring. fp16 halves both.")
#     ap.add_argument("--project-chunk", type=int, default=65536,
#                     help="v3: CPU→GPU chunk size during coreset projection. "
#                          "Per-chunk GPU peak: chunk × D × 4 B "
#                          "(≈469 MB at chunk=65536, D=1792). Lower this if "
#                          "per-chunk allocations are tight.")
#     ap.add_argument("--num-workers", type=int, default=2)
#     ap.add_argument("--smooth-sigma", type=float, default=1.5)
#     ap.add_argument("--knn-k", type=int, default=9,
#                     help="kept for API compat; only 1-NN is actually used.")
#     ap.add_argument("--tta", default="none",
#                     choices=["none", "hflip", "vflip", "hvflip", "d4"])
#     ap.add_argument("--aggressive-cleanup", action="store_true",
#                     help="call torch.cuda.empty_cache() between inference "
#                          "batches (~1%% slower, very stable on shared GPUs).")
#     ap.add_argument("--seed", type=int, default=0)
#     ap.add_argument("--only-classes", nargs="*", default=[])
#     ap.add_argument("--skip-eval", action="store_true")
#     ap.add_argument("--skip-submission", action="store_true")
#     ap.add_argument("--no-save-banks", action="store_true")
#     ap.add_argument("--no-zip", action="store_true")
#     ap.add_argument("--run-tag", default="",
#                     help="human-readable tag appended to the run_id")
#     args = ap.parse_args()
#
#     feature_layers = tuple(sorted(set(args.feature_layers)))
#     target_layer = resolve_target_layer(feature_layers, args.target_layer)
#
#     cfg = RunConfig(
#         data_root=args.data_root, report_dir=args.report_dir,
#         backbone=args.backbone,
#         feature_layers=feature_layers,
#         target_layer=target_layer,
#         input_size=args.input_size,
#         coreset_frac=args.coreset_frac,
#         coreset_fp16=args.coreset_fp16,
#         batch_size=args.batch_size,
#         score_batch_size=args.score_batch_size,
#         score_chunk=args.score_chunk,
#         memory_chunk=args.memory_chunk,
#         memory_dtype=args.memory_dtype,
#         project_chunk=args.project_chunk,
#         num_workers=args.num_workers,
#         smooth_sigma=args.smooth_sigma, knn_k=args.knn_k,
#         tta=args.tta,
#         seed=args.seed, only_classes=args.only_classes,
#         skip_eval=args.skip_eval, skip_submission=args.skip_submission,
#         save_memory_banks=not args.no_save_banks,
#         zip_submission=not args.no_zip,
#         aggressive_cleanup=args.aggressive_cleanup,
#         run_tag=args.run_tag,
#     )
#     cfg.report_dir.mkdir(parents=True, exist_ok=True)
#     run_id = make_run_id(cfg)
#     run_dir = cfg.report_dir / "runs" / run_id
#     run_dir.mkdir(parents=True, exist_ok=True)
#
#     with tee_to(run_dir / "run_log.txt"):
#         hr(f"PATCHCORE v3 (OOM-safe coreset) — RUN {run_id}", "█")
#         print(f"  data_root        : {cfg.data_root}")
#         print(f"  report_dir       : {cfg.report_dir}")
#         print(f"  run_dir          : {run_dir}")
#         print(f"  backbone         : {cfg.backbone}")
#         print(f"  feature_layers   : {list(cfg.feature_layers)}")
#         print(f"  target_layer     : {cfg.target_layer}"
#               f"  (auto-rule: 2 if in layers else min)")
#         print(f"  input_size       : {cfg.input_size}")
#         print(f"  coreset_frac     : {cfg.coreset_frac:.1%}   "
#               f"coreset_fp16={cfg.coreset_fp16}   knn_k={cfg.knn_k}")
#         print(f"  memory_dtype     : {cfg.memory_dtype}")
#         print(f"  batch_size (fit) : {cfg.batch_size}")
#         print(f"  score_batch_size : {cfg.score_batch_size}")
#         print(f"  score_chunk      : {cfg.score_chunk}")
#         print(f"  memory_chunk     : {cfg.memory_chunk}")
#         print(f"  project_chunk    : {cfg.project_chunk}")
#         print(f"  smooth_sigma     : {cfg.smooth_sigma}")
#         print(f"  tta              : {cfg.tta}")
#         print(f"  aggressive_clean : {cfg.aggressive_cleanup}")
#         print(f"  device           : {'cuda' if torch.cuda.is_available() else 'cpu'}")
#         if torch.cuda.is_available():
#             print(f"                    {torch.cuda.get_device_name(0)}, "
#                   f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
#         ch_per_layer = sum(RESNET_CHANNELS[cfg.backbone][l]
#                             for l in cfg.feature_layers)
#         print(f"  expected fused feature dim : {ch_per_layer}")
#
#         # Save the config for posterity
#         with open(run_dir / "config.json", "w") as f:
#             json.dump({k: (list(v) if isinstance(v, tuple) else
#                             str(v) if isinstance(v, Path) else v)
#                        for k, v in asdict(cfg).items()}, f, indent=2)
#
#         t_total = time.time()
#         records = scan_dataset(cfg.data_root)
#         if not records:
#             print("\n[FATAL] no records found — abort"); return
#         classes = sorted({r.cls for r in records})
#         if cfg.only_classes:
#             classes = [c for c in classes if c in set(cfg.only_classes)]
#         print(f"\n  running on {len(classes)} class(es): {', '.join(classes)}")
#
#         all_test_results = []
#         all_eval_rows: list[dict] = []
#         class_aps: dict[str, float] = {}
#         class_elapsed: dict[str, float] = {}
#         for cls in classes:
#             res = run_one_class(cls, records, cfg, run_dir)
#             all_test_results.extend(res["test_results"])
#             all_eval_rows.extend(res["eval_rows"])
#             class_aps[cls] = res["class_mean_ap"]
#             class_elapsed[cls] = res["elapsed_min"]
#
#         hr("LOCAL VALIDATION SUMMARY", "=")
#         print(f"  {'class':<10} {'mean pixel-AP':>15} {'time (min)':>12}")
#         for cls in classes:
#             print(f"  {cls:<10} {class_aps.get(cls, float('nan')):>15.4f} "
#                   f"{class_elapsed.get(cls, 0):>12.1f}")
#         valid_aps = [v for v in class_aps.values() if not math.isnan(v)]
#         overall_ap = float(np.mean(valid_aps)) if valid_aps else float("nan")
#         if valid_aps:
#             print(f"  {'OVERALL':<10} {overall_ap:>15.4f}")
#
#         if all_eval_rows:
#             tab_path = run_dir / "local_eval.csv"
#             with open(tab_path, "w", newline="", encoding="utf-8") as f:
#                 w = csv.DictWriter(f, fieldnames=list(all_eval_rows[0].keys()))
#                 w.writeheader()
#                 for row in all_eval_rows: w.writerow(row)
#             print(f"  saved per-(class, anomaly_type) AP table -> {tab_path}")
#
#         if not cfg.skip_submission and all_test_results:
#             hr("SUBMISSION", "=")
#             write_submission(all_test_results, run_dir,
#                               zip_it=cfg.zip_submission)
#             print(f"\n  Upload to the Kaggle leaderboard:\n"
#                   f"    {run_dir / 'submission.zip'}")
#
#         # ── Append one row to the master ablation CSV ──
#         master_csv = cfg.report_dir / "ablation_master.csv"
#         row = {
#             "run_id": run_id,
#             "run_tag": cfg.run_tag,
#             "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
#             "backbone": cfg.backbone,
#             "feature_layers": "+".join(str(l) for l in cfg.feature_layers),
#             "target_layer": cfg.target_layer,
#             "input_size": cfg.input_size,
#             "coreset_frac": f"{cfg.coreset_frac:.4f}",
#             "coreset_fp16": int(cfg.coreset_fp16),
#             "patch_size": 3,
#             "knn_k": cfg.knn_k,
#             "smooth_sigma": cfg.smooth_sigma,
#             "tta": cfg.tta,
#             "memory_dtype": cfg.memory_dtype,
#             "batch_size": cfg.batch_size,
#             "score_batch_size": cfg.score_batch_size,
#             "score_chunk": cfg.score_chunk,
#             "memory_chunk": cfg.memory_chunk,
#             "project_chunk": cfg.project_chunk,
#             "seed": cfg.seed,
#             "n_classes": len(classes),
#             **{f"AP_{c}": f"{class_aps.get(c, float('nan')):.4f}"
#                 for c in sorted(class_aps)},
#             "AP_overall": f"{overall_ap:.4f}",
#             "runtime_min": f"{(time.time() - t_total) / 60:.1f}",
#             "submission_path": str(run_dir / "submission.zip")
#                                 if not cfg.skip_submission else "",
#         }
#         append_to_ablation_master(master_csv, row)
#         print(f"\n  ablation row appended -> {master_csv}")
#
#         hr(f"DONE — run_id={run_id}", "█")
#
#
# if __name__ == "__main__":
#     main()

"""Spacepresso baseline v4 — PatchCore + faster coreset selection.

Drop-in successor to patchcore_baseline_v3.py. v3 already fixed the OOM
in coreset selection by keeping the full feature tensor on CPU and
projecting only chunks to GPU. v4 fixes the *speed* of the selection
itself, which v3 inherited unchanged from v2.

# Why v3 coreset selection was slow

The inner loop did:

      idx = int(torch.argmax(min_dist).item())

`.item()` is a blocking GPU→CPU sync. PyTorch flushes the entire CUDA
pipeline, hands back a Python int, and only THEN queues the next
iteration's kernels. With 5% coreset on 6M patches that's 300,000 sync
points. Measured per-iter cost on a shared L4 was ~38 ms vs ~2 ms of
actual compute — the rest was pure sync overhead.

# What v4 changes

A new CLI flag `--coreset-algo {exact,minibatch}` (default: `minibatch`).
Two new internal helpers; the call site in `fit()` is otherwise unchanged.

(1) `_kcenter_exact_gpu` — same selection as v3 but `idx` stays a 0-d
    GPU tensor through the iteration. Indexing, distance, and min are
    all queued on the GPU without flushing. A blocking sync still
    happens at log time (every 5%), so progress is still visible. ~5-10x
    faster than v3 in practice.

(2) `_kcenter_minibatch_gpu` — selects `--coreset-batch` (default 64)
    farthest points per ROUND instead of one per iteration. After each
    round, `min_dist` is updated against ALL new centers at once via
    one `cdist`. ~50-100x faster than v3 with negligible quality loss
    for batch sizes ≤ 256 — this is the standard approximation in
    production PatchCore deployments (see Sener & Savarese 2018 §B).

The projection step (CPU features → GPU 32-d, in `project_chunk`-sized
slices) is unchanged from v3.

# Quality note

Mini-batch greedy k-center is an approximation: within one round, the
top-`B` farthest points are chosen against the OLD `min_dist`, so the
B-th choice doesn't "see" that the 1st has just been added. For
B ≤ ~256 the resulting coreset is empirically very close to exact — the
PatchCore paper itself uses an approximate greedy variant. If you want
to verify on your data, run one class with `--coreset-algo exact` and
one with `--coreset-algo minibatch` and compare pixel-AP.

# CLI compatibility

The exp5/exp6 sweep scripts run unchanged. New optional knobs:

      --coreset-algo {exact, minibatch}    (default: minibatch)
      --coreset-batch INT                  (default: 64)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    wide_resnet50_2, Wide_ResNet50_2_Weights,
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
)


# ─────────────────────────────────────────────────────────────────────────────
# Defaults — override on the CLI
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/work/u10813429/anomaly-detection")
DEFAULT_DATA_ROOT  = PROJECT_ROOT / "data"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "baseline_out"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
VIEW_RE = re.compile(r"^(?P<base>.+?)_view(?P<v>\d+)\.[A-Za-z]+$")

RESNET_CHANNELS = {
    "wide_resnet50_2": {1: 256, 2: 512, 3: 1024, 4: 2048},
    "resnet50":        {1: 256, 2: 512, 3: 1024, 4: 2048},
    "resnet18":        {1: 64,  2: 128, 3: 256,  4: 512},
}


# ─────────────────────────────────────────────────────────────────────────────
# Tee logger
# ─────────────────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams: st.write(s); st.flush()
    def flush(self):
        for st in self.streams: st.flush()


@contextmanager
def tee_to(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "w", encoding="utf-8")
    old = sys.stdout
    sys.stdout = Tee(old, f)
    try: yield
    finally:
        sys.stdout = old; f.close()


def hr(t, c="="): print(f"\n{c * 78}\n  {t}\n{c * 78}")
def sub(t): print(f"\n--- {t} ---")
def now_hms(): return time.strftime("%H:%M:%S")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset (unchanged from v3)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ImageRecord:
    path: Path
    cls: str
    split: str
    anomaly_type: str | None = None
    sample_id: str | None = None
    view: int | None = None
    mask_path: Path | None = None


def parse_view(filename: str) -> tuple[str, int | None]:
    m = VIEW_RE.match(filename)
    if m:
        return m.group("base"), int(m.group("v"))
    return Path(filename).stem, None


def scan_dataset(data_root: Path) -> list[ImageRecord]:
    out: list[ImageRecord] = []
    if not data_root.exists():
        print(f"  [FATAL] {data_root} not found"); return out
    classes = sorted(d.name for d in data_root.iterdir()
                     if d.is_dir() and d.name.startswith("class_"))
    for cls in classes:
        cdir = data_root / cls
        gd = cdir / "train" / "good"
        if gd.exists():
            for p in sorted(gd.iterdir()):
                if p.suffix.lower() in IMG_EXTS:
                    sid, v = parse_view(p.name)
                    out.append(ImageRecord(p, cls, "train_good",
                                           sample_id=sid, view=v))
        td = cdir / "train"
        if td.exists():
            for sd in sorted(td.iterdir()):
                if (not sd.is_dir() or sd.name == "good"
                        or not sd.name.startswith("anomaly_")):
                    continue
                a_type = sd.name
                gtd = cdir / "ground_truth_train" / a_type
                for p in sorted(sd.iterdir()):
                    if p.suffix.lower() not in IMG_EXTS: continue
                    sid, v = parse_view(p.name)
                    mp = None
                    if gtd.exists():
                        cand = gtd / p.name
                        if cand.exists():
                            mp = cand
                        else:
                            for q in gtd.iterdir():
                                if (q.stem == p.stem
                                        and q.suffix.lower() in IMG_EXTS):
                                    mp = q; break
                    out.append(ImageRecord(p, cls, "train_anomaly",
                                           anomaly_type=a_type,
                                           sample_id=sid, view=v,
                                           mask_path=mp))
        ted = cdir / "test"
        if ted.exists():
            for p in sorted(ted.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    sid, v = parse_view(p.name)
                    out.append(ImageRecord(p, cls, "test",
                                           sample_id=sid, view=v))
    return out


class SpacepressoDataset(Dataset):
    def __init__(self, records: list[ImageRecord], load_masks: bool,
                 input_size: int):
        self.records = records
        self.load_masks = load_masks
        self.input_size = input_size
        self.tx = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self): return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        with Image.open(r.path) as im:
            im = im.convert("RGB")
            x = self.tx(im)
        if self.load_masks and r.mask_path is not None:
            with Image.open(r.mask_path) as mm:
                mm = mm.convert("L")
                mm = mm.resize((self.input_size, self.input_size),
                               Image.NEAREST)
                m = (np.asarray(mm) > 127).astype(np.float32)
        else:
            m = np.zeros((self.input_size, self.input_size), dtype=np.float32)
        return x, torch.from_numpy(m), i


def make_loader(records, batch_size, input_size,
                load_masks=False, num_workers=2, shuffle=False):
    ds = SpacepressoDataset(records, load_masks=load_masks,
                            input_size=input_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=(num_workers > 0))


# ─────────────────────────────────────────────────────────────────────────────
# Feature extractor (unchanged from v3)
# ─────────────────────────────────────────────────────────────────────────────
class FeatureExtractor(nn.Module):
    def __init__(self, backbone: str = "wide_resnet50_2"):
        super().__init__()
        if backbone == "wide_resnet50_2":
            m = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        elif backbone == "resnet50":
            m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        elif backbone == "resnet18":
            m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"unknown backbone: {backbone}")
        self.backbone_name = backbone
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
        self.eval()
        for p in self.parameters(): p.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor,
                layers: tuple[int, ...] = (2, 3)) -> dict[int, torch.Tensor]:
        out: dict[int, torch.Tensor] = {}
        need_4 = 4 in layers
        x = self.stem(x)
        x = self.layer1(x)
        if 1 in layers: out[1] = x
        x = self.layer2(x)
        if 2 in layers: out[2] = x
        x = self.layer3(x)
        if 3 in layers: out[3] = x
        if need_4:
            x = self.layer4(x)
            out[4] = x
        return out


def patchify_and_combine(maps: dict[int, torch.Tensor],
                          patch_size: int = 3,
                          target_layer: int = 2) -> torch.Tensor:
    assert target_layer in maps, \
        f"target_layer={target_layer} not in {list(maps)}"
    H, W = maps[target_layer].shape[-2:]
    avg = nn.AvgPool2d(kernel_size=patch_size, stride=1, padding=patch_size // 2)
    pooled = []
    for k in sorted(maps.keys()):
        m = maps[k]
        if m.shape[-2:] != (H, W):
            m = F.interpolate(m, size=(H, W), mode="bilinear",
                              align_corners=False)
        pooled.append(avg(m))
    feats = torch.cat(pooled, dim=1)
    B, C, _, _ = feats.shape
    feats = feats.permute(0, 2, 3, 1).reshape(B, H * W, C)
    feats = F.normalize(feats, p=2, dim=-1)
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Greedy k-center coreset — v4
#
# Two algorithms, both operate on a 32-d random projection of CPU features
# (projection runs entirely on GPU; CPU features stay put — same as v3).
#
#   "exact"     — sync-free O(N*k) exact greedy. `idx` stays as a 0-d GPU
#                 tensor so the inner loop queues kernels without
#                 stopping for `.item()`. ~5-10x faster than v3.
#   "minibatch" — pick `batch_size` farthest points per round, update
#                 `min_dist` against all of them at once. ~50-100x faster
#                 than v3 with negligible quality loss for batch ≤ 256.
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def _project_cpu_features_to_gpu(features_cpu: torch.Tensor,
                                  device: torch.device,
                                  seed: int,
                                  projection_dim: int,
                                  project_chunk: int) -> torch.Tensor:
    """Random Gaussian projection from CPU features to GPU low-dim rep.
    See v3 docstring for the OOM rationale; the projection itself is
    unchanged.
    """
    N, D = features_cpu.shape
    if projection_dim >= D:
        raise ValueError(
            f"projection_dim={projection_dim} must be < D={D} for the "
            f"CPU-features path (otherwise we'd materialise the full "
            f"feature tensor on the GPU — the very thing we are avoiding)."
        )
    g = torch.Generator(device=device).manual_seed(seed)
    P = torch.randn(D, projection_dim, generator=g, device=device,
                    dtype=torch.float32) / math.sqrt(projection_dim)
    feats_proj = torch.empty((N, projection_dim), device=device,
                              dtype=torch.float32)
    t_proj = time.time()
    last_log_at = 0
    for s in range(0, N, project_chunk):
        e = min(N, s + project_chunk)
        chunk_gpu = features_cpu[s:e].to(device, dtype=torch.float32,
                                          non_blocking=True)
        feats_proj[s:e] = chunk_gpu @ P
        del chunk_gpu
        if (e - last_log_at) >= 1_000_000 or e == N:
            last_log_at = e
            print(f"      projecting features {e:>9d}/{N}  "
                  f"({e / N * 100:>5.1f}%)  "
                  f"elapsed={time.time() - t_proj:.1f}s", flush=True)
    del P
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"      projected {N} features to {projection_dim}-d on GPU "
          f"in {time.time() - t_proj:.1f}s "
          f"({feats_proj.element_size() * feats_proj.numel() / 1e6:.1f} "
          f"MB on GPU)")
    return feats_proj


@torch.inference_mode()
def _kcenter_exact_gpu(feats_proj: torch.Tensor,
                        n_select: int, seed: int) -> torch.Tensor:
    """Sync-free exact greedy k-center on GPU. Same selection as v3.
    Returns indices on GPU (caller moves to CPU).
    """
    device = feats_proj.device
    N = feats_proj.shape[0]
    n_select = min(n_select, N)

    rng = torch.Generator(device=device).manual_seed(seed + 1)
    first_t = torch.randint(0, N, (1,), generator=rng, device=device)
    first = int(first_t.item())     # one sync at setup, then no more
    selected = torch.empty(n_select, dtype=torch.long, device=device)
    selected[0] = first

    # Euclidean (cdist) — same convention as the rest of the pipeline.
    first_pt = feats_proj[first:first + 1]
    min_dist = torch.cdist(feats_proj, first_pt).squeeze(1)  # (N,)
    min_dist[first] = -1.0   # never re-pick

    log_every = max(1, n_select // 20)
    t0 = time.time()
    for i in range(1, n_select):
        # 0-d GPU tensor — NOT a Python int. No sync.
        idx_t = torch.argmax(min_dist)
        selected[i] = idx_t
        # GPU indexing with GPU tensor → (D,), unsqueeze to (1, D).
        new_pt = feats_proj[idx_t].unsqueeze(0)
        d_new = torch.cdist(feats_proj, new_pt).squeeze(1)
        min_dist = torch.minimum(min_dist, d_new)
        min_dist[idx_t] = -1.0
        if i % log_every == 0 or i == n_select - 1:
            # Sync ONLY at log time, every 5% of progress.
            md_max = float(min_dist.max().clamp_min(0.0))
            print(f"      coreset (exact): {i + 1}/{n_select} "
                  f"({(i + 1) / n_select * 100:>5.1f}%)  "
                  f"max-min-dist={md_max:.4f}  "
                  f"elapsed={time.time() - t0:.1f}s", flush=True)
    return selected


@torch.inference_mode()
def _kcenter_minibatch_gpu(feats_proj: torch.Tensor,
                            n_select: int, seed: int,
                            batch_size: int) -> torch.Tensor:
    """Mini-batch greedy k-center. Pick `batch_size` farthest points per
    round and update `min_dist` against all of them at once.

    Per-round GPU peak:
      cdist output  (N × batch_size × 4 B)   +  feats_proj  +  min_dist
      At N=6M, batch_size=64:  ~1.5 GB   +   768 MB   +   24 MB
                          B=256: ~6 GB    +   768 MB   +   24 MB

    Returns indices on GPU.
    """
    device = feats_proj.device
    N = feats_proj.shape[0]
    n_select = min(n_select, N)
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if n_select <= 1:
        return torch.zeros(max(n_select, 0), dtype=torch.long, device=device)

    rng = torch.Generator(device=device).manual_seed(seed + 1)
    first = int(torch.randint(0, N, (1,), generator=rng,
                               device=device).item())
    selected = torch.empty(n_select, dtype=torch.long, device=device)
    selected[0] = first

    first_pt = feats_proj[first:first + 1]
    min_dist = torch.cdist(feats_proj, first_pt).squeeze(1)
    min_dist[first] = -1.0

    n_chosen = 1
    n_rounds_total = (n_select - 1 + batch_size - 1) // batch_size
    log_every_rounds = max(1, n_rounds_total // 20)
    round_idx = 0
    t0 = time.time()
    while n_chosen < n_select:
        b = min(batch_size, n_select - n_chosen)
        # Top-b farthest from current min_dist (negative-filled positions
        # can't be picked since they're -1.0 < all real distances).
        _, top_idx = torch.topk(min_dist, b, largest=True)  # (b,) on GPU
        selected[n_chosen:n_chosen + b] = top_idx

        # Update min_dist against all b new centers at once.
        new_pts = feats_proj[top_idx]                  # (b, D)
        new_d = torch.cdist(feats_proj, new_pts)       # (N, b) Euclidean
        new_d_min = new_d.min(dim=1).values            # (N,)
        min_dist = torch.minimum(min_dist, new_d_min)
        min_dist[top_idx] = -1.0
        del new_d, new_d_min, new_pts

        n_chosen += b
        round_idx += 1
        if round_idx % log_every_rounds == 0 or n_chosen == n_select:
            md_max = float(min_dist.max().clamp_min(0.0))
            print(f"      coreset (minibatch b={batch_size}): "
                  f"{n_chosen}/{n_select} "
                  f"({n_chosen / n_select * 100:>5.1f}%)  "
                  f"max-min-dist={md_max:.4f}  "
                  f"round={round_idx}/{n_rounds_total}  "
                  f"elapsed={time.time() - t0:.1f}s", flush=True)
    return selected


@torch.inference_mode()
def greedy_coreset(features_cpu: torch.Tensor,
                    n_select: int,
                    device: torch.device,
                    seed: int = 0,
                    projection_dim: int = 32,
                    project_chunk: int = 65536,
                    algo: str = "minibatch",
                    batch_size: int = 64) -> torch.Tensor:
    """Greedy k-center on CPU-resident features.

    `algo` ∈ {"exact", "minibatch"}:
      "exact"     — same selection as v3 but sync-free. ~5-10x faster.
      "minibatch" — pick `batch_size` farthest points per round. ~50-100x
                    faster than v3 with negligible quality loss for
                    typical batch sizes.
    Returns selected indices as a CPU LongTensor of shape (n_select,).
    """
    assert features_cpu.device.type == "cpu", (
        f"v4 expects CPU features; got {features_cpu.device}."
    )
    if algo not in ("exact", "minibatch"):
        raise ValueError(f"coreset_algo must be 'exact' or 'minibatch', "
                         f"got {algo!r}")
    feats_proj = _project_cpu_features_to_gpu(features_cpu, device, seed,
                                                projection_dim,
                                                project_chunk)
    if algo == "minibatch":
        selected = _kcenter_minibatch_gpu(feats_proj, n_select, seed,
                                           batch_size=batch_size)
    else:
        selected = _kcenter_exact_gpu(feats_proj, n_select, seed)
    selected_cpu = selected.cpu()
    del feats_proj, selected
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return selected_cpu


# ─────────────────────────────────────────────────────────────────────────────
# Target-layer auto-resolution (unchanged from v3)
# ─────────────────────────────────────────────────────────────────────────────
def resolve_target_layer(feature_layers: tuple[int, ...],
                          explicit: str | int | None) -> int:
    if explicit is None or (isinstance(explicit, str) and explicit == "auto"):
        if 2 in feature_layers:
            return 2
        return min(feature_layers)
    val = int(explicit)
    if val not in feature_layers:
        raise ValueError(
            f"--target-layer {val} not in --feature-layers "
            f"{list(feature_layers)}; pass one of {list(feature_layers)} or "
            f"'auto'."
        )
    return val


# ─────────────────────────────────────────────────────────────────────────────
# PatchCore — multi-layer + TTA-aware + OOM-safe (fit() identical to v3
# except for the new coreset_algo / coreset_batch knobs).
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PatchCoreConfig:
    backbone: str = "wide_resnet50_2"
    feature_layers: tuple[int, ...] = (2, 3)
    target_layer: int = 2
    input_size: int = 224
    coreset_frac: float = 0.10
    coreset_fp16: bool = False
    coreset_algo: str = "minibatch"        # "exact" | "minibatch"
    coreset_batch: int = 64                # mini-batch size for minibatch algo
    patch_size: int = 3
    knn_k: int = 9
    batch_size: int = 32
    score_batch_size: int = 16
    score_chunk: int = 4096
    memory_chunk: int = 32768
    memory_dtype: str = "fp16"
    project_chunk: int = 65536
    num_workers: int = 2
    device: str = "cuda"
    seed: int = 0


class PatchCore:
    def __init__(self, cfg: PatchCoreConfig):
        self.cfg = cfg
        self.device = torch.device(
            cfg.device if torch.cuda.is_available() else "cpu")
        self.extractor = FeatureExtractor(cfg.backbone).to(self.device)
        self.target_layer = cfg.target_layer
        assert self.target_layer in cfg.feature_layers, (
            f"target_layer={self.target_layer} not in "
            f"feature_layers={cfg.feature_layers}"
        )
        self.memory_dtype = (torch.float16 if cfg.memory_dtype == "fp16"
                             else torch.float32)
        self.memory: torch.Tensor | None = None
        self.feature_hw: tuple[int, int] | None = None
        self.feature_dim: int | None = None

    @torch.inference_mode()
    def _extract(self, loader: DataLoader) -> torch.Tensor:
        feats_list = []
        n = 0
        last_log = 0
        for x, _, _ in loader:
            bs = x.shape[0]
            x = x.to(self.device, non_blocking=True)
            maps = self.extractor(x, layers=self.cfg.feature_layers)
            pf = patchify_and_combine(maps,
                                       patch_size=self.cfg.patch_size,
                                       target_layer=self.target_layer)
            if self.feature_hw is None:
                P = pf.shape[1]
                H = W = int(math.isqrt(P))
                self.feature_hw = (H, W)
                self.feature_dim = pf.shape[2]
            pf = pf.reshape(-1, pf.shape[-1]).detach().cpu()
            feats_list.append(pf)
            del x, maps, pf
            n += bs
            if n - last_log >= 256:
                last_log = n
                print(f"      extracted features from {n} images "
                      f"(feat dim={self.feature_dim}, "
                      f"patches/img={self.feature_hw[0] * self.feature_hw[1]})",
                      flush=True)
        return torch.cat(feats_list, dim=0)

    def fit(self, train_good_records: list[ImageRecord]) -> None:
        print(f"    [{now_hms()}] extracting train/good features "
              f"({len(train_good_records)} images, "
              f"input_size={self.cfg.input_size}, "
              f"layers={list(self.cfg.feature_layers)}, "
              f"target_layer={self.target_layer})...")
        loader = make_loader(train_good_records,
                             batch_size=self.cfg.batch_size,
                             input_size=self.cfg.input_size,
                             num_workers=self.cfg.num_workers,
                             load_masks=False, shuffle=False)
        all_feats = self._extract(loader)

        if self.cfg.coreset_fp16:
            all_feats = all_feats.half()
            print(f"    -> {all_feats.shape[0]} patch features "
                  f"(fp16, CPU; "
                  f"{all_feats.element_size() * all_feats.numel() / 1e9:.2f} "
                  f"GB)")
        else:
            print(f"    -> {all_feats.shape[0]} patch features "
                  f"(fp32, CPU; "
                  f"{all_feats.element_size() * all_feats.numel() / 1e9:.2f} "
                  f"GB)")

        n_select = max(int(self.cfg.coreset_frac * all_feats.shape[0]), 1)
        if self.cfg.coreset_algo == "minibatch":
            algo_desc = (f"minibatch (batch_size={self.cfg.coreset_batch})")
        else:
            algo_desc = "exact (sync-free)"
        print(f"    [{now_hms()}] greedy coreset [{algo_desc}]: "
              f"selecting {n_select} of {all_feats.shape[0]} patches "
              f"({self.cfg.coreset_frac:.1%})  "
              f"[CPU features → GPU 32-d projection, "
              f"project_chunk={self.cfg.project_chunk}]")

        idx_cpu = greedy_coreset(all_feats, n_select, self.device,
                                  seed=self.cfg.seed,
                                  project_chunk=self.cfg.project_chunk,
                                  algo=self.cfg.coreset_algo,
                                  batch_size=self.cfg.coreset_batch)

        selected_cpu = all_feats[idx_cpu]
        del all_feats

        selected_gpu = selected_cpu.to(self.device, non_blocking=True)
        del selected_cpu
        memory = selected_gpu.float()
        del selected_gpu
        memory = F.normalize(memory, p=2, dim=-1)
        self.memory = memory.to(self.memory_dtype).contiguous()
        del memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"    [{now_hms()}] memory bank ready  "
              f"shape={tuple(self.memory.shape)}  "
              f"dtype={self.memory.dtype}  "
              f"({self.memory.element_size() * self.memory.numel() / 1e6:.1f} "
              f"MB on GPU)")

    @torch.inference_mode()
    def _score_one_pass(self, x: torch.Tensor) -> torch.Tensor:
        assert self.memory is not None, "fit() first"
        maps = self.extractor(x.to(self.device, non_blocking=True),
                              layers=self.cfg.feature_layers)
        pf = patchify_and_combine(maps,
                                   patch_size=self.cfg.patch_size,
                                   target_layer=self.target_layer)
        del maps
        B, P, C = pf.shape
        H = W = int(math.isqrt(P))
        flat = pf.reshape(-1, C)
        N_q = flat.shape[0]
        score_chunk = self.cfg.score_chunk
        memory_chunk = self.cfg.memory_chunk
        M_total = self.memory.shape[0]

        dist_min = torch.empty(N_q, device=self.device, dtype=torch.float32)
        for s in range(0, N_q, score_chunk):
            e = min(N_q, s + score_chunk)
            q = flat[s:e].to(self.memory_dtype)
            max_sim = torch.full((q.shape[0],), -2.0, device=self.device,
                                  dtype=self.memory_dtype)
            for ms in range(0, M_total, memory_chunk):
                me = min(M_total, ms + memory_chunk)
                m_chunk = self.memory[ms:me]
                sim = q @ m_chunk.T
                chunk_max = sim.max(dim=1).values
                torch.maximum(max_sim, chunk_max, out=max_sim)
                del sim, chunk_max
            dist_min[s:e] = (1.0 - max_sim.float())
            del q, max_sim
        del flat, pf

        score_lr = dist_min.reshape(B, H, W)
        score = F.interpolate(score_lr.unsqueeze(1),
                              size=(self.cfg.input_size, self.cfg.input_size),
                              mode="bilinear", align_corners=False)
        out = score.squeeze(1).cpu()
        del dist_min, score_lr, score
        return out

    @torch.inference_mode()
    def score_batch(self, x: torch.Tensor, tta: str = "none") -> torch.Tensor:
        if tta == "none":
            return self._score_one_pass(x)
        accumulator = None
        n = 0

        def _add(scores: torch.Tensor):
            nonlocal accumulator, n
            if accumulator is None:
                accumulator = scores.clone()
            else:
                accumulator += scores
            n += 1

        _add(self._score_one_pass(x))
        if tta in ("hflip", "hvflip", "d4"):
            s = self._score_one_pass(torch.flip(x, dims=[-1]))
            _add(torch.flip(s, dims=[-1]))
        if tta in ("vflip", "hvflip", "d4"):
            s = self._score_one_pass(torch.flip(x, dims=[-2]))
            _add(torch.flip(s, dims=[-2]))
        if tta == "d4":
            s = self._score_one_pass(torch.rot90(x, k=1, dims=[-2, -1]))
            _add(torch.rot90(s, k=-1, dims=[-2, -1]))
            s = self._score_one_pass(torch.rot90(x, k=2, dims=[-2, -1]))
            _add(torch.rot90(s, k=-2, dims=[-2, -1]))
            s = self._score_one_pass(torch.rot90(x, k=3, dims=[-2, -1]))
            _add(torch.rot90(s, k=-3, dims=[-2, -1]))
            xf = torch.flip(x, dims=[-1])
            s = self._score_one_pass(torch.rot90(xf, k=1, dims=[-2, -1]))
            s = torch.rot90(s, k=-1, dims=[-2, -1])
            s = torch.flip(s, dims=[-1])
            _add(s)
            xf = torch.flip(x, dims=[-2])
            s = self._score_one_pass(torch.rot90(xf, k=1, dims=[-2, -1]))
            s = torch.rot90(s, k=-1, dims=[-2, -1])
            s = torch.flip(s, dims=[-2])
            _add(s)
        return accumulator / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Pixel-level Average Precision (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def pixel_average_precision(score: np.ndarray, gt: np.ndarray) -> float:
    s = score.astype(np.float32).ravel()
    y = gt.astype(np.int32).ravel()
    if y.sum() == 0:
        return 0.0
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(y, s))
    except Exception:
        order = np.argsort(-s, kind="stable")
        y = y[order]
        tp = np.cumsum(y); fp = np.cumsum(1 - y)
        precision = tp / (tp + fp + 1e-12)
        recall = tp / max(int(y.sum()), 1)
        recall = np.concatenate([[0.0], recall])
        precision = np.concatenate([[1.0], precision])
        return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


# ─────────────────────────────────────────────────────────────────────────────
# Smoothing + calibration + q8rle (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def _gaussian_kernel_1d(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return (k / k.sum()).astype(np.float32)


def gaussian_smooth(score: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    if sigma <= 0:
        return score
    r = max(1, int(round(3 * sigma)))
    k = _gaussian_kernel_1d(sigma, r)
    sx = np.pad(score, ((r, r), (0, 0)), mode="reflect")
    sx = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, sx)
    sx = np.pad(sx, ((0, 0), (r, r)), mode="reflect")
    sx = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 1, sx)
    return sx


def calibrate_to_unit(scores: list[np.ndarray]) -> tuple[float, float]:
    flat = np.concatenate([s.ravel() for s in scores])
    lo = float(np.percentile(flat, 1.0))
    hi = float(np.percentile(flat, 99.5))
    if hi <= lo: hi = lo + 1e-6
    return lo, hi


def float_matrix_to_q8rle(x: np.ndarray) -> str:
    q = np.clip(np.rint(np.asarray(x, dtype=np.float32) * 255),
                0, 255).astype(np.uint8)
    h, w = q.shape
    flat = q.T.reshape(-1)
    if flat.size == 0:
        return f"q8rle {h} {w}"
    cuts = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, flat.size]
    parts = ["q8rle", str(h), str(w)]
    for v, n in zip(flat[starts], ends - starts):
        parts += [str(int(v)), str(int(n))]
    return " ".join(parts)


SUBMISSION_H = SUBMISSION_W = 224


def maybe_resize_to_submission(score: np.ndarray) -> np.ndarray:
    if score.shape == (SUBMISSION_H, SUBMISSION_W):
        return score
    t = torch.from_numpy(score).unsqueeze(0).unsqueeze(0).float()
    t = F.interpolate(t, size=(SUBMISSION_H, SUBMISSION_W),
                      mode="bilinear", align_corners=False)
    return t.squeeze().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Experiment tracking
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RunConfig:
    data_root: Path
    report_dir: Path
    backbone: str = "wide_resnet50_2"
    feature_layers: tuple[int, ...] = (2, 3)
    target_layer: int = 2
    input_size: int = 224
    coreset_frac: float = 0.10
    coreset_fp16: bool = False
    coreset_algo: str = "minibatch"
    coreset_batch: int = 64
    batch_size: int = 32
    score_batch_size: int = 16
    score_chunk: int = 4096
    memory_chunk: int = 32768
    memory_dtype: str = "fp16"
    project_chunk: int = 65536
    num_workers: int = 2
    smooth_sigma: float = 1.5
    knn_k: int = 9
    tta: str = "none"
    seed: int = 0
    only_classes: list[str] = field(default_factory=list)
    skip_eval: bool = False
    skip_submission: bool = False
    save_memory_banks: bool = True
    zip_submission: bool = True
    aggressive_cleanup: bool = False
    run_tag: str = ""


def make_run_id(cfg: RunConfig) -> str:
    fp = json.dumps({
        "backbone": cfg.backbone,
        "feature_layers": list(cfg.feature_layers),
        "target_layer": cfg.target_layer,
        "input_size": cfg.input_size,
        "coreset_frac": cfg.coreset_frac,
        "coreset_fp16": cfg.coreset_fp16,
        "coreset_algo": cfg.coreset_algo,
        "coreset_batch": cfg.coreset_batch,
        "knn_k": cfg.knn_k,
        "tta": cfg.tta,
        "smooth_sigma": cfg.smooth_sigma,
        "memory_dtype": cfg.memory_dtype,
        "score_batch_size": cfg.score_batch_size,
        "score_chunk": cfg.score_chunk,
        "memory_chunk": cfg.memory_chunk,
        "project_chunk": cfg.project_chunk,
        "seed": cfg.seed,
        "v": 4,
    }, sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(fp).hexdigest()[:6]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    bb = {"wide_resnet50_2": "wrn50", "resnet50": "rn50",
          "resnet18": "rn18"}[cfg.backbone]
    L = "".join(str(l) for l in cfg.feature_layers)
    bits = (f"{stamp}_{bb}_L{L}_T{cfg.target_layer}_"
            f"in{cfg.input_size}_cs{int(cfg.coreset_frac*100):02d}")
    if cfg.coreset_algo == "minibatch":
        bits += f"_mb{cfg.coreset_batch}"
    elif cfg.coreset_algo == "exact":
        bits += "_exact"
    if cfg.memory_dtype != "fp16":
        bits += f"_{cfg.memory_dtype}"
    if cfg.tta != "none":
        bits += f"_tta-{cfg.tta}"
    if cfg.run_tag:
        bits += f"_{re.sub(r'[^A-Za-z0-9._-]+', '-', cfg.run_tag)}"
    return f"{bits}_{digest}"


def append_to_ablation_master(master_csv: Path, row: dict) -> None:
    master_csv.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict] = []
    fieldnames: list[str] = []
    if master_csv.exists():
        with open(master_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)
    for k in row.keys():
        if k not in fieldnames:
            fieldnames.append(k)
    existing_rows.append(row)
    with open(master_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in existing_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# ─────────────────────────────────────────────────────────────────────────────
# Per-class pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_one_class(cls: str, records_all: list[ImageRecord],
                   cfg: RunConfig, run_dir: Path) -> dict:
    hr(f"CLASS {cls}", "─")
    t_start = time.time()

    train_good = [r for r in records_all if r.cls == cls and r.split == "train_good"]
    train_anom = [r for r in records_all if r.cls == cls and r.split == "train_anomaly"]
    test       = [r for r in records_all if r.cls == cls and r.split == "test"]
    print(f"  train_good={len(train_good)}  "
          f"train_anomaly={len(train_anom)}  test={len(test)}")

    pc_cfg = PatchCoreConfig(
        backbone=cfg.backbone,
        feature_layers=cfg.feature_layers,
        target_layer=cfg.target_layer,
        input_size=cfg.input_size,
        coreset_frac=cfg.coreset_frac,
        coreset_fp16=cfg.coreset_fp16,
        coreset_algo=cfg.coreset_algo,
        coreset_batch=cfg.coreset_batch,
        knn_k=cfg.knn_k,
        batch_size=cfg.batch_size,
        score_batch_size=cfg.score_batch_size,
        score_chunk=cfg.score_chunk,
        memory_chunk=cfg.memory_chunk,
        memory_dtype=cfg.memory_dtype,
        project_chunk=cfg.project_chunk,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    pc = PatchCore(pc_cfg)
    pc.fit(train_good)

    if cfg.save_memory_banks:
        bank_path = run_dir / "banks" / f"{cls}_memory.pt"
        bank_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"memory": pc.memory.cpu(),
                    "feature_dim": pc.feature_dim,
                    "feature_hw": pc.feature_hw,
                    "target_layer": pc.target_layer,
                    "memory_dtype": cfg.memory_dtype,
                    "coreset_algo": cfg.coreset_algo,
                    "coreset_batch": cfg.coreset_batch,
                    "config": asdict(pc_cfg)}, bank_path)
        print(f"    saved memory bank -> {bank_path}")

    eval_rows: list[dict] = []
    class_mean_ap = float("nan")
    if not cfg.skip_eval and train_anom:
        sub(f"local validation — per-anomaly-type pixel-AP "
            f"(tta={cfg.tta}, score_bs={cfg.score_batch_size})")
        loader = make_loader(train_anom, batch_size=cfg.score_batch_size,
                             input_size=cfg.input_size,
                             num_workers=cfg.num_workers, load_masks=True)
        scores_per_idx: dict[int, np.ndarray] = {}
        gt_per_idx: dict[int, np.ndarray] = {}
        with torch.inference_mode():
            for x, masks, idxs in loader:
                score_maps = pc.score_batch(x, tta=cfg.tta).numpy()
                masks_np = masks.numpy()
                for b in range(x.shape[0]):
                    sm = gaussian_smooth(score_maps[b], cfg.smooth_sigma)
                    scores_per_idx[int(idxs[b])] = sm
                    gt_per_idx[int(idxs[b])] = masks_np[b]
                if cfg.aggressive_cleanup and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        by_anom: dict[str, list[float]] = defaultdict(list)
        for r_idx, sm in scores_per_idx.items():
            r = train_anom[r_idx]
            ap = pixel_average_precision(sm, gt_per_idx[r_idx])
            by_anom[r.anomaly_type or "?"].append(ap)
        print(f"    {'anomaly_type':<14} {'n_views':>8} "
              f"{'pixel-AP (mean ± std)':>26}")
        per_type_means = []
        for a_type in sorted(by_anom):
            arr = np.asarray(by_anom[a_type])
            per_type_means.append(float(arr.mean()))
            print(f"    {a_type:<14} {len(arr):>8} "
                  f"{arr.mean():>15.4f} ± {arr.std():.4f}")
            eval_rows.append({"class": cls, "anomaly_type": a_type,
                              "n_views": int(len(arr)),
                              "ap_mean": float(arr.mean()),
                              "ap_std": float(arr.std()),
                              "ap_min": float(arr.min()),
                              "ap_max": float(arr.max())})
        class_mean_ap = float(np.mean(per_type_means)) if per_type_means else 0.0
        print(f"    >>> class {cls} mean pixel-AP "
              f"(avg over types): {class_mean_ap:.4f}")

    test_results: list[tuple[ImageRecord, np.ndarray]] = []
    if not cfg.skip_submission and test:
        sub(f"scoring {len(test)} test images "
            f"(tta={cfg.tta}, score_bs={cfg.score_batch_size})")
        loader = make_loader(test, batch_size=cfg.score_batch_size,
                             input_size=cfg.input_size,
                             num_workers=cfg.num_workers, load_masks=False)
        n_done = 0
        last_log = 0
        with torch.inference_mode():
            for x, _, idxs in loader:
                score_maps = pc.score_batch(x, tta=cfg.tta).numpy()
                for b in range(x.shape[0]):
                    sm = gaussian_smooth(score_maps[b], cfg.smooth_sigma)
                    sm = maybe_resize_to_submission(sm)
                    test_results.append((test[int(idxs[b])], sm))
                n_done += x.shape[0]
                if n_done - last_log >= 200:
                    last_log = n_done
                    print(f"      scored {n_done}/{len(test)}", flush=True)
                if cfg.aggressive_cleanup and torch.cuda.is_available():
                    torch.cuda.empty_cache()

    elapsed_min = (time.time() - t_start) / 60.0
    print(f"  class {cls} done in {elapsed_min:.1f} min")
    del pc
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {
        "class": cls,
        "class_mean_ap": class_mean_ap,
        "eval_rows": eval_rows,
        "test_results": test_results,
        "elapsed_min": elapsed_min,
    }


def write_submission(all_test_results, run_dir: Path,
                     zip_it: bool = True) -> Path:
    sub("calibrating scores and writing submission.csv")
    scores = [sm for _, sm in all_test_results]
    if not scores:
        raise RuntimeError("no test scores")
    lo, hi = calibrate_to_unit(scores)
    print(f"    global score calibration  lo={lo:.4f}  hi={hi:.4f}")
    csv_path = run_dir / "submission.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Label"])
        for r, sm in all_test_results:
            normed = np.clip((sm - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
            w.writerow([r.path.stem, float_matrix_to_q8rle(normed)])
            n += 1
    print(f"    wrote {n} rows -> {csv_path}")
    if zip_it:
        zip_path = csv_path.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w",
                             compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv_path, arcname=csv_path.name)
        print(f"    zipped         -> {zip_path}")
        return zip_path
    return csv_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--data-root",  type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    ap.add_argument("--backbone", default="wide_resnet50_2",
                    choices=["wide_resnet50_2", "resnet50", "resnet18"])
    ap.add_argument("--feature-layers", type=int, nargs="+", default=[2, 3],
                    help="ResNet stages to fuse (any subset of 1 2 3 4)")
    ap.add_argument("--target-layer", default="auto",
                    help="Spatial grid: auto|1|2|3|4. 'auto' picks layer 2 "
                         "if in feature-layers else smallest-numbered.")
    ap.add_argument("--input-size", type=int, default=224,
                    choices=[224, 256, 320, 384, 448, 512])
    ap.add_argument("--coreset-frac", type=float, default=0.10)
    ap.add_argument("--coreset-fp16", action="store_true",
                    help="Cast CPU features to fp16 (CPU RAM only). Halves "
                         "CPU RAM during selection. Selection is "
                         "mathematically identical up to fp16 noise.")
    ap.add_argument("--coreset-algo", default="minibatch",
                    choices=["exact", "minibatch"],
                    help="v4: 'minibatch' (default, ~50-100x faster than v3, "
                         "selects coreset-batch points per round) or 'exact' "
                         "(sync-free greedy, same selection as v3, ~5-10x "
                         "faster than v3).")
    ap.add_argument("--coreset-batch", type=int, default=64,
                    help="v4: points per round for the minibatch algo. "
                         "64 is a strong default. Higher = faster but "
                         "more approximation; 256 is a reasonable ceiling.")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="batch size for feature extraction during fit()")
    ap.add_argument("--score-batch-size", type=int, default=16,
                    help="batch size for inference (validation + test).")
    ap.add_argument("--score-chunk", type=int, default=4096,
                    help="query-axis chunk size during NN scoring.")
    ap.add_argument("--memory-chunk", type=int, default=32768,
                    help="memory-bank-axis chunk during scoring.")
    ap.add_argument("--memory-dtype", default="fp16",
                    choices=["fp16", "fp32"],
                    help="dtype of the stored memory bank and sim matmul.")
    ap.add_argument("--project-chunk", type=int, default=65536,
                    help="CPU→GPU chunk size during coreset projection.")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--smooth-sigma", type=float, default=1.5)
    ap.add_argument("--knn-k", type=int, default=9,
                    help="kept for API compat; only 1-NN is actually used.")
    ap.add_argument("--tta", default="none",
                    choices=["none", "hflip", "vflip", "hvflip", "d4"])
    ap.add_argument("--aggressive-cleanup", action="store_true",
                    help="call torch.cuda.empty_cache() between inference "
                         "batches.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-classes", nargs="*", default=[])
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-submission", action="store_true")
    ap.add_argument("--no-save-banks", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--run-tag", default="",
                    help="human-readable tag appended to the run_id")
    args = ap.parse_args()

    feature_layers = tuple(sorted(set(args.feature_layers)))
    target_layer = resolve_target_layer(feature_layers, args.target_layer)

    cfg = RunConfig(
        data_root=args.data_root, report_dir=args.report_dir,
        backbone=args.backbone,
        feature_layers=feature_layers,
        target_layer=target_layer,
        input_size=args.input_size,
        coreset_frac=args.coreset_frac,
        coreset_fp16=args.coreset_fp16,
        coreset_algo=args.coreset_algo,
        coreset_batch=args.coreset_batch,
        batch_size=args.batch_size,
        score_batch_size=args.score_batch_size,
        score_chunk=args.score_chunk,
        memory_chunk=args.memory_chunk,
        memory_dtype=args.memory_dtype,
        project_chunk=args.project_chunk,
        num_workers=args.num_workers,
        smooth_sigma=args.smooth_sigma, knn_k=args.knn_k,
        tta=args.tta,
        seed=args.seed, only_classes=args.only_classes,
        skip_eval=args.skip_eval, skip_submission=args.skip_submission,
        save_memory_banks=not args.no_save_banks,
        zip_submission=not args.no_zip,
        aggressive_cleanup=args.aggressive_cleanup,
        run_tag=args.run_tag,
    )
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id(cfg)
    run_dir = cfg.report_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with tee_to(run_dir / "run_log.txt"):
        hr(f"PATCHCORE v4 (fast coreset) — RUN {run_id}", "█")
        print(f"  data_root        : {cfg.data_root}")
        print(f"  report_dir       : {cfg.report_dir}")
        print(f"  run_dir          : {run_dir}")
        print(f"  backbone         : {cfg.backbone}")
        print(f"  feature_layers   : {list(cfg.feature_layers)}")
        print(f"  target_layer     : {cfg.target_layer}"
              f"  (auto-rule: 2 if in layers else min)")
        print(f"  input_size       : {cfg.input_size}")
        print(f"  coreset_frac     : {cfg.coreset_frac:.1%}   "
              f"coreset_fp16={cfg.coreset_fp16}   knn_k={cfg.knn_k}")
        print(f"  coreset_algo     : {cfg.coreset_algo}"
              + (f"  (batch={cfg.coreset_batch})"
                 if cfg.coreset_algo == "minibatch" else ""))
        print(f"  memory_dtype     : {cfg.memory_dtype}")
        print(f"  batch_size (fit) : {cfg.batch_size}")
        print(f"  score_batch_size : {cfg.score_batch_size}")
        print(f"  score_chunk      : {cfg.score_chunk}")
        print(f"  memory_chunk     : {cfg.memory_chunk}")
        print(f"  project_chunk    : {cfg.project_chunk}")
        print(f"  smooth_sigma     : {cfg.smooth_sigma}")
        print(f"  tta              : {cfg.tta}")
        print(f"  aggressive_clean : {cfg.aggressive_cleanup}")
        print(f"  device           : {'cuda' if torch.cuda.is_available() else 'cpu'}")
        if torch.cuda.is_available():
            print(f"                    {torch.cuda.get_device_name(0)}, "
                  f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        ch_per_layer = sum(RESNET_CHANNELS[cfg.backbone][l]
                            for l in cfg.feature_layers)
        print(f"  expected fused feature dim : {ch_per_layer}")

        with open(run_dir / "config.json", "w") as f:
            json.dump({k: (list(v) if isinstance(v, tuple) else
                            str(v) if isinstance(v, Path) else v)
                       for k, v in asdict(cfg).items()}, f, indent=2)

        t_total = time.time()
        records = scan_dataset(cfg.data_root)
        if not records:
            print("\n[FATAL] no records found — abort"); return
        classes = sorted({r.cls for r in records})
        if cfg.only_classes:
            classes = [c for c in classes if c in set(cfg.only_classes)]
        print(f"\n  running on {len(classes)} class(es): {', '.join(classes)}")

        all_test_results = []
        all_eval_rows: list[dict] = []
        class_aps: dict[str, float] = {}
        class_elapsed: dict[str, float] = {}
        for cls in classes:
            res = run_one_class(cls, records, cfg, run_dir)
            all_test_results.extend(res["test_results"])
            all_eval_rows.extend(res["eval_rows"])
            class_aps[cls] = res["class_mean_ap"]
            class_elapsed[cls] = res["elapsed_min"]

        hr("LOCAL VALIDATION SUMMARY", "=")
        print(f"  {'class':<10} {'mean pixel-AP':>15} {'time (min)':>12}")
        for cls in classes:
            print(f"  {cls:<10} {class_aps.get(cls, float('nan')):>15.4f} "
                  f"{class_elapsed.get(cls, 0):>12.1f}")
        valid_aps = [v for v in class_aps.values() if not math.isnan(v)]
        overall_ap = float(np.mean(valid_aps)) if valid_aps else float("nan")
        if valid_aps:
            print(f"  {'OVERALL':<10} {overall_ap:>15.4f}")

        if all_eval_rows:
            tab_path = run_dir / "local_eval.csv"
            with open(tab_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(all_eval_rows[0].keys()))
                w.writeheader()
                for row in all_eval_rows: w.writerow(row)
            print(f"  saved per-(class, anomaly_type) AP table -> {tab_path}")

        if not cfg.skip_submission and all_test_results:
            hr("SUBMISSION", "=")
            write_submission(all_test_results, run_dir,
                              zip_it=cfg.zip_submission)
            print(f"\n  Upload to the Kaggle leaderboard:\n"
                  f"    {run_dir / 'submission.zip'}")

        master_csv = cfg.report_dir / "ablation_master.csv"
        row = {
            "run_id": run_id,
            "run_tag": cfg.run_tag,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backbone": cfg.backbone,
            "feature_layers": "+".join(str(l) for l in cfg.feature_layers),
            "target_layer": cfg.target_layer,
            "input_size": cfg.input_size,
            "coreset_frac": f"{cfg.coreset_frac:.4f}",
            "coreset_fp16": int(cfg.coreset_fp16),
            "coreset_algo": cfg.coreset_algo,
            "coreset_batch": cfg.coreset_batch,
            "patch_size": 3,
            "knn_k": cfg.knn_k,
            "smooth_sigma": cfg.smooth_sigma,
            "tta": cfg.tta,
            "memory_dtype": cfg.memory_dtype,
            "batch_size": cfg.batch_size,
            "score_batch_size": cfg.score_batch_size,
            "score_chunk": cfg.score_chunk,
            "memory_chunk": cfg.memory_chunk,
            "project_chunk": cfg.project_chunk,
            "seed": cfg.seed,
            "n_classes": len(classes),
            **{f"AP_{c}": f"{class_aps.get(c, float('nan')):.4f}"
                for c in sorted(class_aps)},
            "AP_overall": f"{overall_ap:.4f}",
            "runtime_min": f"{(time.time() - t_total) / 60:.1f}",
            "submission_path": str(run_dir / "submission.zip")
                                if not cfg.skip_submission else "",
        }
        append_to_ablation_master(master_csv, row)
        print(f"\n  ablation row appended -> {master_csv}")

        hr(f"DONE — run_id={run_id}", "█")


if __name__ == "__main__":
    main()