# """TextAD — Text-Conditioned Synthetic Anomaly Detection.
#
# # Concept
#
#   Spacepresso class notes 16–19/05 ("Synthetic Scars", "Inpainting the
#   Defect", "Dreaming Image and Mask", "Language as a Lantern") point at
#   a single idea: train a per-pixel segmentation network on synthetic
#   (image, mask) pairs whose defects MATCH the real defect taxonomy of
#   each class. The taxonomy is read at runtime from
#   data/anomaly_descriptions.csv — language is the lantern that tells us
#   which defects to forge.
#
#   defect_library.py implements the lantern and the synthesis. This file
#   implements the segmentation U-Net, the per-class training loop, and
#   the stacker integration (submission.csv + local_predictions.npz +
#   ablation_master row, with image_paths preserved so the v4/v5 stacker
#   can build cross-view features).
#
# # Why this is different from DRAEM (your other synthetic-defect baseline)
#
#   DRAEM uses a class-agnostic Perlin-mask + random-colour-texture
#   scheme that's the same for every class. TextAD pulls the defect
#   taxonomy per class from the dataset's descriptions, so class_06
#   (coffee) trains on mold + contamination defects, class_04 (screw)
#   trains on scratches + dents + fragments, and so on. Same training
#   budget, different generative prior — orthogonal signal for the stacker.
#
#   Architectural difference: DRAEM trains TWO U-Nets (reconstructive R +
#   discriminative D, ~63M params at base=32). TextAD trains ONE seg
#   U-Net (~7M params at base=32, 4 levels). Per-class training fits in
#   ~3 min on an L4; full 8-class run ~25 min.
#
# # Stacker contract (xgboost_stacker_v5)
#
#   This file writes the SAME three artefacts every other baseline writes:
#
#       $RUN/submission.csv          ← test scores (q8rle, calibrated to [0, 1])
#       $RUN/local_predictions.npz   ← per-pixel val scores + GTs + image_paths
#       <ablation_master.csv row>    ← appended with backbone='TEXTAD'
#
#   Critically, LocalPredSaver.add() is called with image_path=r.path so
#   the v4 stacker's PATH_VIEW_RE can parse _viewN out of the filename
#   and build cross-view features (xv_max/xv_mean/xv_std/xv_lonely, CVD,
#   per-(class, view) rank-norm). Without image_path those features are
#   zero for TextAD.
#
#   The run_dir name starts with "textad_" so a one-line addition to
#   MODEL_FAMILY_PATTERNS in xgboost_stacker_v5.py picks it up as its
#   own family. Without that addition the family resolves to "unknown"
#   which also defaults to small_cc=0 — same end behaviour.
#
# # Memory/speed on L4 24 GB
#
#   Per class @ input 256, bs=16, 2500 iters, base=32:
#     Training: ~3 min   (one U-Net, AMP)
#     Eval:     ~10 s
#     Test:     ~25 s
#   Full 8-class run: ~25 min.
#
# # Dependencies
#
#   defect_library.py and patchcore_baseline_v2.py (scan_dataset,
#   gaussian_smooth, etc.) and local_preds_saver.py in same directory.
# """
# from __future__ import annotations
#
# import argparse
# import csv
# import hashlib
# import json
# import math
# import random
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
#
# sys.path.insert(0, str(Path(__file__).resolve().parent))
# from patchcore_baseline_v2 import (
#     ImageRecord, scan_dataset,
#     pixel_average_precision, gaussian_smooth,
#     calibrate_to_unit, float_matrix_to_q8rle,
#     maybe_resize_to_submission,
#     append_to_ablation_master,
# )
# from local_preds_saver import LocalPredSaver
# from defect_library import (
#     DefectSpec, load_class_taxonomy,
#     make_class_defect_specs, inject_defects,
# )
#
#
# PROJECT_ROOT       = Path("/work/u10813429/anomaly-detection")
# DEFAULT_DATA_ROOT  = PROJECT_ROOT / "data"
# DEFAULT_REPORT_DIR = PROJECT_ROOT / "baseline_out"
# DEFAULT_DESCRIPTIONS_CSV = DEFAULT_DATA_ROOT / "anomaly_descriptions.csv"
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
# # Datasets
# #
# # Augmentation diversity: per-worker advancing Generator (same pattern
# # as the DRAEM v1.1 fix). Without this, every (worker, idx) pair gets
# # the same Perlin/scratch every epoch.
# # ─────────────────────────────────────────────────────────────────────────────
# _WORKER_RNG: np.random.Generator | None = None
#
#
# def worker_init_fn(_worker_id):
#     global _WORKER_RNG
#     base = torch.initial_seed() % (2 ** 32)
#     np.random.seed(base)
#     random.seed(base)
#     _WORKER_RNG = np.random.default_rng(base)
#
#
# class TextadTrainDataset(Dataset):
#     """train/good images with synthetic class-specific defects.
#
#     Returns:
#       img  : (3, H, W) float32 in [0, 1]      — corrupted (or clean)
#       mask : (H, W)    float32 in {0, 1}      — defect mask
#     """
#     def __init__(self, records: list[ImageRecord], input_size: int,
#                  defect_specs: list[DefectSpec],
#                  anomaly_prob: float = 0.7,
#                  n_defects_range: tuple[int, int] = (1, 3),
#                  seed: int = 0):
#         self.records = records
#         self.input_size = input_size
#         self.defect_specs = defect_specs
#         self.anomaly_prob = anomaly_prob
#         self.n_defects_range = n_defects_range
#         self.seed = seed
#         # No ImageNet norm — keep [0, 1] (matches DRAEM and is friendlier
#         # for the synthesis math).
#         self.resize = transforms.Resize((input_size, input_size))
#
#     def __len__(self): return len(self.records)
#
#     def _load_rgb01(self, path: Path) -> np.ndarray:
#         with Image.open(path) as im:
#             im = im.convert("RGB")
#             im = self.resize(im)
#             return np.asarray(im, dtype=np.float32) / 255.0
#
#     def _get_rng(self, item_idx: int) -> np.random.Generator:
#         global _WORKER_RNG
#         if _WORKER_RNG is None:
#             seed = (self.seed * 1_000_003 + item_idx * 9973) & 0xFFFFFFFF
#             _WORKER_RNG = np.random.default_rng(seed)
#         return _WORKER_RNG
#
#     def __getitem__(self, i):
#         r = self.records[i]
#         rng = self._get_rng(i)
#         img = self._load_rgb01(r.path)
#         if rng.random() < self.anomaly_prob and self.defect_specs:
#             img, mask = inject_defects(
#                 img, self.defect_specs, rng,
#                 n_defects_range=self.n_defects_range)
#         else:
#             mask = np.zeros(img.shape[:2], dtype=np.float32)
#         img_t  = torch.from_numpy(img).permute(2, 0, 1).contiguous()
#         mask_t = torch.from_numpy(mask).contiguous()
#         return img_t, mask_t
#
#
# class TextadInferenceDataset(Dataset):
#     """test/train_anomaly images. Returns (img01, mask_or_zeros, index)."""
#     def __init__(self, records: list[ImageRecord], input_size: int,
#                  load_masks: bool):
#         self.records = records
#         self.input_size = input_size
#         self.load_masks = load_masks
#         self.resize = transforms.Resize((input_size, input_size))
#
#     def __len__(self): return len(self.records)
#
#     def __getitem__(self, i):
#         r = self.records[i]
#         with Image.open(r.path) as im:
#             im = im.convert("RGB")
#             im = self.resize(im)
#             arr = np.asarray(im, dtype=np.float32) / 255.0
#         x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
#         if self.load_masks and r.mask_path is not None:
#             with Image.open(r.mask_path) as mm:
#                 mm = mm.convert("L").resize(
#                     (self.input_size, self.input_size), Image.NEAREST)
#                 m = (np.asarray(mm) > 127).astype(np.float32)
#         else:
#             m = np.zeros((self.input_size, self.input_size),
#                           dtype=np.float32)
#         return x, torch.from_numpy(m), i
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Segmentation U-Net — 4-level, GroupNorm, ~7M params at base=32
# # ─────────────────────────────────────────────────────────────────────────────
# def _conv_block(c_in: int, c_out: int) -> nn.Module:
#     return nn.Sequential(
#         nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
#         nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
#         nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out),
#         nn.ReLU(inplace=True),
#     )
#
#
# class _Down(nn.Module):
#     def __init__(self, c_in, c_out):
#         super().__init__()
#         self.pool = nn.MaxPool2d(2)
#         self.conv = _conv_block(c_in, c_out)
#     def forward(self, x): return self.conv(self.pool(x))
#
#
# class _Up(nn.Module):
#     def __init__(self, c_in, c_skip, c_out):
#         super().__init__()
#         self.up = nn.Upsample(scale_factor=2, mode="bilinear",
#                                 align_corners=False)
#         self.conv = _conv_block(c_in + c_skip, c_out)
#     def forward(self, x, skip):
#         x = self.up(x)
#         if x.shape[-2:] != skip.shape[-2:]:
#             x = F.interpolate(x, size=skip.shape[-2:],
#                               mode="bilinear", align_corners=False)
#         return self.conv(torch.cat([x, skip], dim=1))
#
#
# class SegUNet(nn.Module):
#     """4-level U-Net. Input/output spatial size identical. Requires
#     input_size divisible by 16 (four 2× downsamples)."""
#     def __init__(self, in_channels: int = 3, out_channels: int = 2,
#                  base: int = 32):
#         super().__init__()
#         c = base
#         self.inc = _conv_block(in_channels, c)
#         self.d1  = _Down(c,      c * 2)
#         self.d2  = _Down(c * 2,  c * 4)
#         self.d3  = _Down(c * 4,  c * 8)
#         self.d4  = _Down(c * 8,  c * 16)
#         self.u4  = _Up(c * 16, c * 8, c * 8)
#         self.u3  = _Up(c * 8,  c * 4, c * 4)
#         self.u2  = _Up(c * 4,  c * 2, c * 2)
#         self.u1  = _Up(c * 2,  c,     c)
#         self.outc = nn.Conv2d(c, out_channels, 1)
#
#     def forward(self, x):
#         e0 = self.inc(x)
#         e1 = self.d1(e0)
#         e2 = self.d2(e1)
#         e3 = self.d3(e2)
#         b  = self.d4(e3)
#         d3 = self.u4(b,  e3)
#         d2 = self.u3(d3, e2)
#         d1 = self.u2(d2, e1)
#         d0 = self.u1(d1, e0)
#         return self.outc(d0)
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Loss
# # ─────────────────────────────────────────────────────────────────────────────
# def focal_loss(logits: torch.Tensor, target: torch.Tensor,
#                 gamma: float = 2.0, alpha: float = 0.5) -> torch.Tensor:
#     """Multi-class focal loss, identical to DRAEM's. logits: (B, 2, H, W);
#     target: (B, H, W) with 0/1 class indices. `alpha` weights class 1."""
#     logp = F.log_softmax(logits, dim=1)
#     p = logp.exp()
#     target = target.long()
#     logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
#     p_t    = p.gather(1, target.unsqueeze(1)).squeeze(1)
#     w = torch.where(target == 1, torch.full_like(p_t, alpha),
#                                   torch.full_like(p_t, 1.0 - alpha))
#     loss = -w * ((1.0 - p_t) ** gamma) * logp_t
#     return loss.mean()
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Training
# # ─────────────────────────────────────────────────────────────────────────────
# def train_textad(net: SegUNet, records: list[ImageRecord],
#                   cfg: "RunConfig", device: torch.device,
#                   specs: list[DefectSpec]) -> None:
#     ds = TextadTrainDataset(
#         records, input_size=cfg.input_size,
#         defect_specs=specs,
#         anomaly_prob=cfg.anomaly_prob,
#         n_defects_range=tuple(cfg.n_defects),
#         seed=cfg.seed)
#     loader = DataLoader(
#         ds, batch_size=cfg.batch_size, shuffle=True,
#         num_workers=cfg.num_workers, pin_memory=True,
#         persistent_workers=(cfg.num_workers > 0),
#         worker_init_fn=worker_init_fn, drop_last=True)
#     iters_per_epoch = max(len(loader), 1)
#     if cfg.total_iters and cfg.total_iters > 0:
#         n_epochs = max(1, math.ceil(cfg.total_iters / iters_per_epoch))
#         print(f"    [auto-epoch] total_iters={cfg.total_iters} / "
#               f"{iters_per_epoch} iters/epoch -> {n_epochs} epochs "
#               f"(--epochs={cfg.epochs} overridden)")
#     else:
#         n_epochs = cfg.epochs
#     total_iters = iters_per_epoch * n_epochs
#
#     optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr,
#                                     weight_decay=cfg.weight_decay)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer, T_max=max(total_iters, 1))
#     use_amp = (device.type == "cuda" and cfg.amp)
#     scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
#
#     fam_summary = ", ".join(s.family for s in specs)
#     print(f"    [{now_hms()}] training: {n_epochs} epochs × "
#           f"{iters_per_epoch} iters ({total_iters} total) "
#           f"bs={cfg.batch_size} amp={use_amp} anom_prob={cfg.anomaly_prob}")
#     print(f"      defect families : {fam_summary}")
#     log_every = max(1, n_epochs // 8)
#     t0 = time.time()
#     for epoch in range(n_epochs):
#         net.train()
#         loss_sum = 0.0
#         n = 0
#         for img, mask in loader:
#             img  = img.to(device, non_blocking=True)
#             mask = mask.to(device, non_blocking=True)
#             img  = torch.nan_to_num(img,  nan=0.0, posinf=1.0, neginf=0.0)
#             mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
#
#             optimizer.zero_grad(set_to_none=True)
#             with torch.amp.autocast("cuda", enabled=use_amp):
#                 logits = net(img)
#                 loss = focal_loss(logits, mask,
#                                     gamma=cfg.focal_gamma,
#                                     alpha=cfg.focal_alpha)
#
#             # AMP grad-clip ordering: backward → unscale → clip → step → update
#             scaler.scale(loss).backward()
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
#             scaler.step(optimizer)
#             scaler.update()
#             scheduler.step()
#
#             B = img.shape[0]
#             loss_sum += loss.item() * B
#             n += B
#         if (epoch + 1) % log_every == 0 or epoch == n_epochs - 1:
#             print(f"      epoch {epoch+1:>3}/{n_epochs}  "
#                   f"loss={loss_sum/max(n,1):.4f}  "
#                   f"lr={scheduler.get_last_lr()[0]:.2e}  "
#                   f"elapsed={time.time()-t0:.1f}s", flush=True)
#     net.eval()
#     print(f"    [{now_hms()}] training done ({time.time()-t0:.1f}s)")
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Inference primitives
# # ─────────────────────────────────────────────────────────────────────────────
# @torch.inference_mode()
# def _score_one_pass(net: SegUNet, x: torch.Tensor,
#                      cfg: "RunConfig") -> torch.Tensor:
#     """Per-pixel anomaly probability at the input resolution.
#     Returns (B, H, W) on the configured device."""
#     use_amp = (cfg.device.type == "cuda" and cfg.amp)
#     with torch.amp.autocast("cuda", enabled=use_amp):
#         logits = net(x)
#     return F.softmax(logits.float(), dim=1)[:, 1]
#
#
# @torch.inference_mode()
# def score_batch(net: SegUNet, x: torch.Tensor,
#                   cfg: "RunConfig") -> torch.Tensor:
#     """TTA-averaged probability on CPU."""
#     x = x.to(cfg.device, non_blocking=True)
#     acc = None; n = 0
#
#     def _add(s):
#         nonlocal acc, n
#         if acc is None: acc = s.clone()
#         else: acc += s
#         n += 1
#
#     _add(_score_one_pass(net, x, cfg))
#     if cfg.tta in ("hflip", "hvflip"):
#         s = _score_one_pass(net, torch.flip(x, dims=[-1]), cfg)
#         _add(torch.flip(s, dims=[-1]))
#     if cfg.tta in ("vflip", "hvflip"):
#         s = _score_one_pass(net, torch.flip(x, dims=[-2]), cfg)
#         _add(torch.flip(s, dims=[-2]))
#     return (acc / max(n, 1)).cpu()
#
#
# def _score_records(net, records, cfg, load_masks):
#     ds = TextadInferenceDataset(records, input_size=cfg.input_size,
#                                   load_masks=load_masks)
#     loader = DataLoader(ds, batch_size=cfg.score_batch_size, shuffle=False,
#                          num_workers=cfg.num_workers, pin_memory=True,
#                          persistent_workers=(cfg.num_workers > 0))
#     scores, gts = {}, {}
#     n_done = 0; last_log = 0
#     for x, masks, idxs in loader:
#         sm = score_batch(net, x, cfg).numpy()
#         m_np = masks.numpy()
#         for b in range(sm.shape[0]):
#             scores[int(idxs[b])] = sm[b]
#             gts[int(idxs[b])] = m_np[b]
#         n_done += sm.shape[0]
#         if n_done - last_log >= 200:
#             last_log = n_done
#             print(f"      scored {n_done}/{len(records)}", flush=True)
#     return scores, gts
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Per-class pipeline
# # ─────────────────────────────────────────────────────────────────────────────
# def run_one_class(cls, records_all, cfg, run_dir, device,
#                    taxonomy, local_saver=None) -> dict:
#     hr(f"CLASS {cls}", "─")
#     t_start = time.time()
#     train_good = [r for r in records_all
#                    if r.cls == cls and r.split == "train_good"]
#     train_anom = [r for r in records_all
#                    if r.cls == cls and r.split == "train_anomaly"]
#     test       = [r for r in records_all
#                    if r.cls == cls and r.split == "test"]
#     print(f"  train_good={len(train_good)}  "
#           f"train_anomaly={len(train_anom)}  test={len(test)}")
#     if not train_good:
#         return {"class": cls, "class_mean_ap": float("nan"),
#                 "eval_rows": [], "test_results": [], "elapsed_min": 0.0}
#
#     specs = make_class_defect_specs(cls, taxonomy)
#     print(f"  defect specs   : {[s.family for s in specs]}")
#
#     net = SegUNet(in_channels=3, out_channels=2,
#                     base=cfg.unet_base).to(device)
#     n_params = sum(p.numel() for p in net.parameters())
#     print(f"  SegUNet: base={cfg.unet_base}  "
#           f"params={n_params/1e6:.2f}M  "
#           f"levels=4 (input must be %16==0)")
#
#     train_textad(net, train_good, cfg, device, specs)
#
#     if cfg.save_checkpoints:
#         ck = run_dir / "ckpt" / f"{cls}_textad.pt"
#         ck.parent.mkdir(parents=True, exist_ok=True)
#         torch.save({"net": net.state_dict(),
#                     "defect_families": [s.family for s in specs]}, ck)
#         print(f"    saved checkpoint -> {ck}")
#
#     eval_rows = []
#     class_mean_ap = float("nan")
#     if not cfg.skip_eval and train_anom:
#         sub(f"local validation  tta={cfg.tta}")
#         scores, gts = _score_records(net, train_anom, cfg, load_masks=True)
#         by_anom = defaultdict(list)
#         for r_idx, sm in scores.items():
#             r = train_anom[r_idx]
#             sm_smooth = gaussian_smooth(sm, cfg.smooth_sigma)
#             ap = pixel_average_precision(sm_smooth, gts[r_idx])
#             by_anom[r.anomaly_type or "?"].append(ap)
#             if local_saver is not None:
#                 # image_path REQUIRED for the v4 stacker's view parsing.
#                 local_saver.add(
#                     cls=cls,
#                     anomaly_type=r.anomaly_type or "unknown",
#                     view_idx=int(r_idx),
#                     score_map=sm_smooth,
#                     gt_mask=gts[r_idx],
#                     image_path=r.path)
#         print(f"    {'anomaly_type':<14} {'n_views':>8} "
#               f"{'pixel-AP (mean ± std)':>26}")
#         per_type_means = []
#         for a_type in sorted(by_anom):
#             arr = np.asarray(by_anom[a_type])
#             per_type_means.append(float(arr.mean()))
#             print(f"    {a_type:<14} {len(arr):>8} "
#                   f"{arr.mean():>15.4f} ± {arr.std():.4f}")
#             eval_rows.append({"class": cls, "anomaly_type": a_type,
#                                "n_views": int(len(arr)),
#                                "ap_mean": float(arr.mean()),
#                                "ap_std": float(arr.std()),
#                                "ap_min": float(arr.min()),
#                                "ap_max": float(arr.max())})
#         class_mean_ap = (float(np.mean(per_type_means))
#                           if per_type_means else 0.0)
#         print(f"    >>> class {cls} mean pixel-AP: {class_mean_ap:.4f}")
#
#     test_results = []
#     if not cfg.skip_submission and test:
#         sub(f"scoring {len(test)} test images  tta={cfg.tta}")
#         scores, _ = _score_records(net, test, cfg, load_masks=False)
#         for r_idx, sm in scores.items():
#             sm_smooth = gaussian_smooth(sm, cfg.smooth_sigma)
#             sm_final = maybe_resize_to_submission(sm_smooth)
#             test_results.append((test[r_idx], sm_final))
#
#     elapsed_min = (time.time() - t_start) / 60.0
#     print(f"  class {cls} done in {elapsed_min:.1f} min")
#     del net
#     if torch.cuda.is_available(): torch.cuda.empty_cache()
#     return {"class": cls, "class_mean_ap": class_mean_ap,
#             "eval_rows": eval_rows, "test_results": test_results,
#             "elapsed_min": elapsed_min}
#
#
# def write_submission(all_test_results, run_dir: Path,
#                       zip_it: bool = True) -> Path:
#     sub("calibrating scores and writing submission.csv")
#     scores = [sm for _, sm in all_test_results]
#     if not scores: raise RuntimeError("no test scores")
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
# # Run config and CLI
# # ─────────────────────────────────────────────────────────────────────────────
# @dataclass
# class RunConfig:
#     data_root: Path
#     report_dir: Path
#     descriptions_csv: Path
#     input_size: int = 256
#     unet_base: int = 32
#     # Training
#     epochs: int = 200
#     total_iters: int | None = 2500
#     batch_size: int = 16
#     lr: float = 1e-4
#     weight_decay: float = 0.0
#     anomaly_prob: float = 0.7
#     n_defects: tuple[int, int] = (1, 3)
#     focal_gamma: float = 2.0
#     focal_alpha: float = 0.5
#     amp: bool = True
#     num_workers: int = 8
#     # Inference
#     score_batch_size: int = 16
#     smooth_sigma: float = 1.5
#     tta: str = "hvflip"
#     # Bookkeeping
#     seed: int = 0
#     only_classes: list[str] = field(default_factory=list)
#     skip_eval: bool = False
#     skip_submission: bool = False
#     save_checkpoints: bool = False
#     zip_submission: bool = True
#     run_tag: str = ""
#     device: torch.device | None = None
#
#
# def make_run_id(cfg: RunConfig) -> str:
#     fp = json.dumps({
#         "method": "textad",
#         "input_size": cfg.input_size,
#         "unet_base": cfg.unet_base,
#         "total_iters": cfg.total_iters,
#         "batch_size": cfg.batch_size,
#         "lr": cfg.lr,
#         "anomaly_prob": cfg.anomaly_prob,
#         "n_defects": list(cfg.n_defects),
#         "focal_gamma": cfg.focal_gamma,
#         "focal_alpha": cfg.focal_alpha,
#         "tta": cfg.tta,
#         "smooth_sigma": cfg.smooth_sigma,
#         "seed": cfg.seed,
#         "v": "1.0",
#     }, sort_keys=True).encode("utf-8")
#     digest = hashlib.sha1(fp).hexdigest()[:6]
#     stamp = time.strftime("%Y%m%d-%H%M%S")
#     budget = (f"it{cfg.total_iters}"
#               if (cfg.total_iters and cfg.total_iters > 0)
#               else f"e{cfg.epochs}")
#     bits = (f"{stamp}_textad_in{cfg.input_size}_b{cfg.unet_base}"
#             f"_{budget}_bs{cfg.batch_size}"
#             f"_nd{cfg.n_defects[0]}-{cfg.n_defects[1]}"
#             f"_p{cfg.anomaly_prob:.2f}")
#     if cfg.tta != "none":
#         bits += f"_tta-{cfg.tta}"
#     if cfg.run_tag:
#         bits += f"_{re.sub(r'[^A-Za-z0-9._-]+', '-', cfg.run_tag)}"
#     return f"{bits}_{digest}"
#
#
# def main():
#     ap = argparse.ArgumentParser(
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         description=__doc__)
#     ap.add_argument("--data-root",  type=Path, default=DEFAULT_DATA_ROOT)
#     ap.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
#     ap.add_argument("--descriptions-csv", type=Path,
#                     default=DEFAULT_DESCRIPTIONS_CSV,
#                     help="Per-class anomaly_descriptions.csv (the lantern). "
#                          "Missing/unreadable file → fallback mixture for "
#                          "every class.")
#     ap.add_argument("--input-size", type=int, default=256,
#                     help="Must be multiple of 16 (four 2x downsamples).")
#     ap.add_argument("--unet-base", type=int, default=32,
#                     help="Base channel count. 32 → ~7M params, "
#                          "24 → ~4M, 48 → ~16M.")
#     # Training
#     ap.add_argument("--epochs", type=int, default=200)
#     ap.add_argument("--total-iters", type=int, default=2500,
#                     help="Overrides --epochs.")
#     ap.add_argument("--batch-size", type=int, default=16)
#     ap.add_argument("--lr", type=float, default=1e-4)
#     ap.add_argument("--weight-decay", type=float, default=0.0)
#     ap.add_argument("--anomaly-prob", type=float, default=0.7,
#                     help="Fraction of training samples that get a synthetic "
#                          "defect. Higher than DRAEM's 0.5 because TextAD's "
#                          "defects are more structured and class-appropriate.")
#     ap.add_argument("--n-defects", type=int, nargs=2, default=[1, 3],
#                     help="Min and max number of defects injected per "
#                          "corrupted training image.")
#     ap.add_argument("--focal-gamma", type=float, default=2.0)
#     ap.add_argument("--focal-alpha", type=float, default=0.5)
#     ap.add_argument("--no-amp", action="store_true")
#     ap.add_argument("--num-workers", type=int, default=8)
#     # Inference
#     ap.add_argument("--score-batch-size", type=int, default=16)
#     ap.add_argument("--smooth-sigma", type=float, default=1.5)
#     ap.add_argument("--tta", default="hvflip",
#                     choices=["none", "hflip", "vflip", "hvflip"])
#     # Bookkeeping
#     ap.add_argument("--seed", type=int, default=0)
#     ap.add_argument("--only-classes", nargs="*", default=[])
#     ap.add_argument("--skip-eval", action="store_true")
#     ap.add_argument("--skip-submission", action="store_true")
#     ap.add_argument("--save-checkpoints", action="store_true")
#     ap.add_argument("--no-zip", action="store_true")
#     ap.add_argument("--no-save-local-preds", action="store_true",
#                     help="Disable saving local_predictions.npz "
#                          "(default: save). Disabling this breaks "
#                          "downstream stacker training.")
#     ap.add_argument("--run-tag", default="")
#     args = ap.parse_args()
#
#     if args.input_size % 16 != 0:
#         raise SystemExit(f"[FATAL] --input-size must be multiple of 16 "
#                           f"(SegUNet has 4 downsamples). "
#                           f"Got {args.input_size}.")
#     if args.n_defects[0] < 1 or args.n_defects[1] < args.n_defects[0]:
#         raise SystemExit(f"[FATAL] --n-defects must be N_min <= N_max and "
#                           f"N_min >= 1. Got {args.n_defects}.")
#
#     cfg = RunConfig(
#         data_root=args.data_root, report_dir=args.report_dir,
#         descriptions_csv=args.descriptions_csv,
#         input_size=args.input_size, unet_base=args.unet_base,
#         epochs=args.epochs, total_iters=args.total_iters,
#         batch_size=args.batch_size, lr=args.lr,
#         weight_decay=args.weight_decay,
#         anomaly_prob=args.anomaly_prob,
#         n_defects=tuple(args.n_defects),
#         focal_gamma=args.focal_gamma, focal_alpha=args.focal_alpha,
#         amp=not args.no_amp, num_workers=args.num_workers,
#         score_batch_size=args.score_batch_size,
#         smooth_sigma=args.smooth_sigma, tta=args.tta,
#         seed=args.seed, only_classes=args.only_classes,
#         skip_eval=args.skip_eval, skip_submission=args.skip_submission,
#         save_checkpoints=args.save_checkpoints,
#         zip_submission=not args.no_zip,
#         run_tag=args.run_tag,
#     )
#     cfg.report_dir.mkdir(parents=True, exist_ok=True)
#     torch.manual_seed(cfg.seed)
#     np.random.seed(cfg.seed)
#     random.seed(cfg.seed)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     cfg.device = device
#
#     run_id = make_run_id(cfg)
#     run_dir = cfg.report_dir / "runs" / run_id
#     run_dir.mkdir(parents=True, exist_ok=True)
#
#     with tee_to(run_dir / "run_log.txt"):
#         hr(f"TEXTAD v1.0 — RUN {run_id}", "█")
#         print(f"  data_root        : {cfg.data_root}")
#         print(f"  run_dir          : {run_dir}")
#         print(f"  descriptions_csv : {cfg.descriptions_csv}")
#         print(f"  input_size       : {cfg.input_size}")
#         print(f"  unet_base        : {cfg.unet_base}")
#         if cfg.total_iters and cfg.total_iters > 0:
#             print(f"  total_iters      : {cfg.total_iters} "
#                   f"(overrides --epochs={cfg.epochs})")
#         else:
#             print(f"  epochs           : {cfg.epochs}")
#         print(f"  batch_size       : {cfg.batch_size}")
#         print(f"  lr / wd          : {cfg.lr} / {cfg.weight_decay}")
#         print(f"  anomaly_prob     : {cfg.anomaly_prob}")
#         print(f"  n_defects        : {cfg.n_defects[0]}–{cfg.n_defects[1]}")
#         print(f"  focal gamma/alpha: {cfg.focal_gamma} / {cfg.focal_alpha}")
#         print(f"  amp              : {cfg.amp}    "
#               f"num_workers: {cfg.num_workers}")
#         print(f"  score_batch_size : {cfg.score_batch_size}")
#         print(f"  smooth_sigma     : {cfg.smooth_sigma}")
#         print(f"  tta              : {cfg.tta}")
#         print(f"  device           : {device}")
#         if torch.cuda.is_available():
#             print(f"                    {torch.cuda.get_device_name(0)}, "
#                   f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
#
#         with open(run_dir / "config.json", "w") as f:
#             cfg_dump = {k: (str(v) if isinstance(v, (Path, torch.device))
#                               else list(v) if isinstance(v, tuple) else v)
#                           for k, v in asdict(cfg).items()}
#             json.dump(cfg_dump, f, indent=2, default=str)
#
#         # Lantern: load per-class taxonomy.
#         sub("loading defect taxonomy from descriptions CSV")
#         taxonomy = load_class_taxonomy(cfg.descriptions_csv)
#         if taxonomy:
#             for c in sorted(taxonomy):
#                 print(f"    {c:<12} → {taxonomy[c]}")
#         else:
#             print(f"    [warn] no taxonomy recovered from "
#                   f"{cfg.descriptions_csv} — using fallback mixture for "
#                   f"every class.")
#
#         t_total = time.time()
#         records = scan_dataset(cfg.data_root)
#         if not records:
#             print("\n[FATAL] no records found"); return
#         classes = sorted({r.cls for r in records})
#         if cfg.only_classes:
#             classes = [c for c in classes if c in set(cfg.only_classes)]
#         print(f"\n  running on {len(classes)} class(es): "
#               f"{', '.join(classes)}")
#
#         local_saver: LocalPredSaver | None = None
#         if not cfg.skip_eval and not args.no_save_local_preds:
#             local_saver = LocalPredSaver()
#
#         all_test_results, all_eval_rows = [], []
#         class_aps, class_elapsed = {}, {}
#         for cls in classes:
#             res = run_one_class(cls, records, cfg, run_dir, device,
#                                   taxonomy=taxonomy,
#                                   local_saver=local_saver)
#             all_test_results.extend(res["test_results"])
#             all_eval_rows.extend(res["eval_rows"])
#             class_aps[cls] = res["class_mean_ap"]
#             class_elapsed[cls] = res["elapsed_min"]
#
#         if local_saver is not None and len(local_saver) > 0:
#             local_saver.save(run_dir / "local_predictions.npz")
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
#                 w = csv.DictWriter(f,
#                                      fieldnames=list(all_eval_rows[0].keys()))
#                 w.writeheader()
#                 for row in all_eval_rows: w.writerow(row)
#             print(f"  saved per-(class, anomaly_type) AP -> {tab_path}")
#
#         if not cfg.skip_submission and all_test_results:
#             hr("SUBMISSION", "=")
#             write_submission(all_test_results, run_dir,
#                               zip_it=cfg.zip_submission)
#             print(f"\n  Upload: {run_dir / 'submission.zip'}")
#
#         master_csv = cfg.report_dir / "ablation_master.csv"
#         row = {
#             "run_id": run_id, "run_tag": cfg.run_tag,
#             "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
#             "backbone": "TEXTAD",
#             "feature_layers": "",
#             "target_layer": "",
#             "input_size": cfg.input_size,
#             "smooth_sigma": cfg.smooth_sigma, "tta": cfg.tta,
#             "batch_size": cfg.batch_size,
#             "score_batch_size": cfg.score_batch_size,
#             "seed": cfg.seed, "n_classes": len(classes),
#             **{f"AP_{c}": f"{class_aps.get(c, float('nan')):.4f}"
#                 for c in sorted(class_aps)},
#             "AP_overall": f"{overall_ap:.4f}",
#             "runtime_min": f"{(time.time() - t_total) / 60:.1f}",
#             "submission_path": str(run_dir / "submission.zip")
#                                 if not cfg.skip_submission else "",
#             "notes": (f"textad v1.0 base{cfg.unet_base} "
#                       f"anom_prob={cfg.anomaly_prob} "
#                       f"nd={cfg.n_defects[0]}-{cfg.n_defects[1]} "
#                       f"{'it' + str(cfg.total_iters) if cfg.total_iters else 'e' + str(cfg.epochs)} "
#                       f"bs{cfg.batch_size}"),
#         }
#         append_to_ablation_master(master_csv, row)
#         print(f"\n  ablation row appended -> {master_csv}")
#         hr(f"DONE — run_id={run_id}", "█")
#
#
# if __name__ == "__main__":
#     main()


"""TextAD — Text-Conditioned Synthetic Anomaly Detection.

# Concept

  Spacepresso class notes 16–19/05 ("Synthetic Scars", "Inpainting the
  Defect", "Dreaming Image and Mask", "Language as a Lantern") point at
  a single idea: train a per-pixel segmentation network on synthetic
  (image, mask) pairs whose defects MATCH the real defect taxonomy of
  each class. The taxonomy is read at runtime from
  data/anomaly_descriptions.csv — language is the lantern that tells us
  which defects to forge.

  defect_library.py implements the lantern and the synthesis. This file
  implements the segmentation U-Net, the per-class training loop, and
  the stacker integration (submission.csv + local_predictions.npz +
  ablation_master row, with image_paths preserved so the v4/v5 stacker
  can build cross-view features).

# Why this is different from DRAEM (your other synthetic-defect baseline)

  DRAEM uses a class-agnostic Perlin-mask + random-colour-texture
  scheme that's the same for every class. TextAD pulls the defect
  taxonomy per class from the dataset's descriptions, so class_06
  (coffee) trains on mold + contamination defects, class_04 (screw)
  trains on scratches + dents + fragments, and so on. Same training
  budget, different generative prior — orthogonal signal for the stacker.

  Architectural difference: DRAEM trains TWO U-Nets (reconstructive R +
  discriminative D, ~63M params at base=32). TextAD trains ONE seg
  U-Net (~7M params at base=32, 4 levels). Per-class training fits in
  ~3 min on an L4; full 8-class run ~25 min.

# Stacker contract (xgboost_stacker_v5)

  This file writes the SAME three artefacts every other baseline writes:

      $RUN/submission.csv          ← test scores (q8rle, calibrated to [0, 1])
      $RUN/local_predictions.npz   ← per-pixel val scores + GTs + image_paths
      <ablation_master.csv row>    ← appended with backbone='TEXTAD'

  Critically, LocalPredSaver.add() is called with image_path=r.path so
  the v4 stacker's PATH_VIEW_RE can parse _viewN out of the filename
  and build cross-view features (xv_max/xv_mean/xv_std/xv_lonely, CVD,
  per-(class, view) rank-norm). Without image_path those features are
  zero for TextAD.

  The run_dir name starts with "textad_" so a one-line addition to
  MODEL_FAMILY_PATTERNS in xgboost_stacker_v5.py picks it up as its
  own family. Without that addition the family resolves to "unknown"
  which also defaults to small_cc=0 — same end behaviour.

# Tier 1 perf changes (vs v1.0)

  * Datasets pre-decode all images to uint8 once in __init__. With
    persistent_workers=True and Linux fork, the cache pages are COW-
    shared with workers (we never mutate), so RAM footprint is ~one
    copy of the cache, not num_workers copies. 2600 train_good images
    × 256² × 3 ≈ 500 MB per class.
  * cudnn.benchmark = True       — picks fastest conv algorithm
    for the fixed input shape (5–15% win on the U-Net).
  * set_float32_matmul_precision("high")  — enables TF32 on Ampere/Ada.
  * net + inputs in channels_last memory format — 10–25% win on Ada
    when paired with AMP.
  * Removed torch.nan_to_num() from the hot loop — the synthetic
    pipeline produces clean tensors, this was defensive overhead.
  * DataLoader prefetch_factor=4 (up from default 2) — covers
    augmentation latency more aggressively.

# Memory/speed on L4 24 GB (v1.1 expected)

  Per class @ input 256, bs=16, 2500 iters, base=32:
    Training: ~20–30 s   (was ~3 min)
    Eval:     ~5 s
    Test:     ~15 s
  Full 8-class run: ~5–8 min (was ~25 min).

# Dependencies

  defect_library.py and patchcore_baseline_v2.py (scan_dataset,
  gaussian_smooth, etc.) and local_preds_saver.py in same directory.
  Requires opencv-python (or opencv-python-headless) — used by
  defect_library.py for fast Gaussian blur.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patchcore_baseline_v2 import (
    ImageRecord, scan_dataset,
    pixel_average_precision, gaussian_smooth,
    calibrate_to_unit, float_matrix_to_q8rle,
    maybe_resize_to_submission,
    append_to_ablation_master,
)
from local_preds_saver import LocalPredSaver
from defect_library import (
    DefectSpec, load_class_taxonomy,
    make_class_defect_specs, inject_defects,
)


PROJECT_ROOT       = Path("/work/u10813429/anomaly-detection")
DEFAULT_DATA_ROOT  = PROJECT_ROOT / "data"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "baseline_out"
DEFAULT_DESCRIPTIONS_CSV = DEFAULT_DATA_ROOT / "anomaly_descriptions.csv"


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


def _build_loader_kwargs(num_workers: int,
                         prefetch_factor: int = 4) -> dict:
    """Build DataLoader kwargs with prefetch_factor only when workers > 0.
    Passing prefetch_factor with num_workers=0 raises in some PyTorch versions."""
    kw = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        kw["prefetch_factor"] = prefetch_factor
    return kw


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
#
# Augmentation diversity: per-worker advancing Generator (same pattern
# as the DRAEM v1.1 fix). Without this, every (worker, idx) pair gets
# the same Perlin/scratch every epoch.
#
# Tier 1 caching: __init__ pre-decodes every image to a uint8 numpy
# array in the main process. DataLoader workers fork (Linux default) and
# inherit the cache as COW-shared memory. Per-epoch disk I/O and PIL
# decode cost drops to zero.
# ─────────────────────────────────────────────────────────────────────────────
_WORKER_RNG: np.random.Generator | None = None


def worker_init_fn(_worker_id):
    global _WORKER_RNG
    base = torch.initial_seed() % (2 ** 32)
    np.random.seed(base)
    random.seed(base)
    _WORKER_RNG = np.random.default_rng(base)


def _decode_rgb_uint8(path: Path, input_size: int) -> np.ndarray:
    """Decode one image to a (H, W, 3) uint8 array at the target size.
    Used by both dataset caches. Bilinear matches torchvision's default."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize(
            (input_size, input_size), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def _decode_mask_uint8(path: Path, input_size: int) -> np.ndarray:
    """Decode a binary mask to a (H, W) uint8 array {0, 1}. Nearest-
    neighbour resize to preserve mask edges."""
    with Image.open(path) as mm:
        mm = mm.convert("L").resize(
            (input_size, input_size), Image.NEAREST)
        return (np.asarray(mm) > 127).astype(np.uint8)


class TextadTrainDataset(Dataset):
    """train/good images with synthetic class-specific defects.

    Returns:
      img  : (3, H, W) float32 in [0, 1]      — corrupted (or clean)
      mask : (H, W)    float32 in {0, 1}      — defect mask

    Tier 1: all images are decoded once in __init__ and held as uint8
    in self._cache. With persistent_workers=True the cache survives
    across epochs; with Linux fork, workers share it COW.
    """
    def __init__(self, records: list[ImageRecord], input_size: int,
                 defect_specs: list[DefectSpec],
                 anomaly_prob: float = 0.7,
                 n_defects_range: tuple[int, int] = (1, 3),
                 seed: int = 0,
                 cache_decoded: bool = True):
        self.records = records
        self.input_size = input_size
        self.defect_specs = defect_specs
        self.anomaly_prob = anomaly_prob
        self.n_defects_range = n_defects_range
        self.seed = seed

        self._cache: list[np.ndarray] | None = None
        if cache_decoded:
            t0 = time.time()
            self._cache = [_decode_rgb_uint8(r.path, input_size)
                            for r in records]
            mb = sum(c.nbytes for c in self._cache) / 1e6
            print(f"    [cache] train: decoded {len(records)} images "
                  f"({mb:.0f} MB, {time.time()-t0:.1f}s)")

    def __len__(self): return len(self.records)

    def _load_rgb01(self, i: int) -> np.ndarray:
        """Return a fresh (H, W, 3) float32 array in [0, 1]. When cached,
        we copy out of uint8 (this is the only per-sample allocation we
        truly cannot avoid)."""
        if self._cache is not None:
            return self._cache[i].astype(np.float32) * (1.0 / 255.0)
        return _decode_rgb_uint8(
            self.records[i].path, self.input_size
        ).astype(np.float32) * (1.0 / 255.0)

    def _get_rng(self, item_idx: int) -> np.random.Generator:
        global _WORKER_RNG
        if _WORKER_RNG is None:
            seed = (self.seed * 1_000_003 + item_idx * 9973) & 0xFFFFFFFF
            _WORKER_RNG = np.random.default_rng(seed)
        return _WORKER_RNG

    def __getitem__(self, i):
        rng = self._get_rng(i)
        img = self._load_rgb01(i)
        if rng.random() < self.anomaly_prob and self.defect_specs:
            img, mask = inject_defects(
                img, self.defect_specs, rng,
                n_defects_range=self.n_defects_range)
        else:
            mask = np.zeros(img.shape[:2], dtype=np.float32)
        img_t  = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        mask_t = torch.from_numpy(mask).contiguous()
        return img_t, mask_t


class TextadInferenceDataset(Dataset):
    """test/train_anomaly images. Returns (img01, mask_or_zeros, index).

    Tier 1: images (and masks, if requested) are pre-decoded into uint8
    caches in __init__."""
    def __init__(self, records: list[ImageRecord], input_size: int,
                 load_masks: bool, cache_decoded: bool = True):
        self.records = records
        self.input_size = input_size
        self.load_masks = load_masks

        self._img_cache: list[np.ndarray] | None = None
        self._mask_cache: list[np.ndarray | None] | None = None

        if cache_decoded:
            t0 = time.time()
            self._img_cache = [_decode_rgb_uint8(r.path, input_size)
                                for r in records]
            mb = sum(c.nbytes for c in self._img_cache) / 1e6
            print(f"    [cache] infer: decoded {len(records)} images "
                  f"({mb:.0f} MB, {time.time()-t0:.1f}s)")
            if load_masks:
                self._mask_cache = []
                for r in records:
                    m: np.ndarray | None = None
                    if r.mask_path is not None:
                        try:
                            m = _decode_mask_uint8(r.mask_path, input_size)
                        except Exception:
                            m = None
                    self._mask_cache.append(m)

        # Pre-allocate a reusable zero mask for records without GT.
        self._zero_mask = np.zeros((input_size, input_size),
                                    dtype=np.float32)

    def __len__(self): return len(self.records)

    def __getitem__(self, i):
        # Image
        if self._img_cache is not None:
            arr = self._img_cache[i].astype(np.float32) * (1.0 / 255.0)
        else:
            arr = (_decode_rgb_uint8(self.records[i].path, self.input_size)
                     .astype(np.float32) * (1.0 / 255.0))
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        # Mask
        if self.load_masks:
            if self._mask_cache is not None:
                m = self._mask_cache[i]
                if m is None:
                    m_np = self._zero_mask
                else:
                    m_np = m.astype(np.float32)
            else:
                r = self.records[i]
                if r.mask_path is not None:
                    try:
                        m_np = _decode_mask_uint8(r.mask_path,
                                                    self.input_size
                                                    ).astype(np.float32)
                    except Exception:
                        m_np = self._zero_mask
                else:
                    m_np = self._zero_mask
        else:
            m_np = self._zero_mask
        return x, torch.from_numpy(m_np), i


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation U-Net — 4-level, GroupNorm, ~7M params at base=32
# ─────────────────────────────────────────────────────────────────────────────
def _conv_block(c_in: int, c_out: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out),
        nn.ReLU(inplace=True),
    )


class _Down(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _conv_block(c_in, c_out)
    def forward(self, x): return self.conv(self.pool(x))


class _Up(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear",
                                align_corners=False)
        self.conv = _conv_block(c_in + c_skip, c_out)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class SegUNet(nn.Module):
    """4-level U-Net. Input/output spatial size identical. Requires
    input_size divisible by 16 (four 2× downsamples)."""
    def __init__(self, in_channels: int = 3, out_channels: int = 2,
                 base: int = 32):
        super().__init__()
        c = base
        self.inc = _conv_block(in_channels, c)
        self.d1  = _Down(c,      c * 2)
        self.d2  = _Down(c * 2,  c * 4)
        self.d3  = _Down(c * 4,  c * 8)
        self.d4  = _Down(c * 8,  c * 16)
        self.u4  = _Up(c * 16, c * 8, c * 8)
        self.u3  = _Up(c * 8,  c * 4, c * 4)
        self.u2  = _Up(c * 4,  c * 2, c * 2)
        self.u1  = _Up(c * 2,  c,     c)
        self.outc = nn.Conv2d(c, out_channels, 1)

    def forward(self, x):
        e0 = self.inc(x)
        e1 = self.d1(e0)
        e2 = self.d2(e1)
        e3 = self.d3(e2)
        b  = self.d4(e3)
        d3 = self.u4(b,  e3)
        d2 = self.u3(d3, e2)
        d1 = self.u2(d2, e1)
        d0 = self.u1(d1, e0)
        return self.outc(d0)


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────
def focal_loss(logits: torch.Tensor, target: torch.Tensor,
                gamma: float = 2.0, alpha: float = 0.5) -> torch.Tensor:
    """Multi-class focal loss, identical to DRAEM's. logits: (B, 2, H, W);
    target: (B, H, W) with 0/1 class indices. `alpha` weights class 1."""
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    target = target.long()
    logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    p_t    = p.gather(1, target.unsqueeze(1)).squeeze(1)
    w = torch.where(target == 1, torch.full_like(p_t, alpha),
                                  torch.full_like(p_t, 1.0 - alpha))
    loss = -w * ((1.0 - p_t) ** gamma) * logp_t
    return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_textad(net: SegUNet, records: list[ImageRecord],
                  cfg: "RunConfig", device: torch.device,
                  specs: list[DefectSpec]) -> None:
    ds = TextadTrainDataset(
        records, input_size=cfg.input_size,
        defect_specs=specs,
        anomaly_prob=cfg.anomaly_prob,
        n_defects_range=tuple(cfg.n_defects),
        seed=cfg.seed)
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True,
        worker_init_fn=worker_init_fn, drop_last=True,
        **_build_loader_kwargs(cfg.num_workers))
    iters_per_epoch = max(len(loader), 1)
    if cfg.total_iters and cfg.total_iters > 0:
        n_epochs = max(1, math.ceil(cfg.total_iters / iters_per_epoch))
        print(f"    [auto-epoch] total_iters={cfg.total_iters} / "
              f"{iters_per_epoch} iters/epoch -> {n_epochs} epochs "
              f"(--epochs={cfg.epochs} overridden)")
    else:
        n_epochs = cfg.epochs
    total_iters = iters_per_epoch * n_epochs

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr,
                                    weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_iters, 1))
    use_amp = (device.type == "cuda" and cfg.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    fam_summary = ", ".join(s.family for s in specs)
    print(f"    [{now_hms()}] training: {n_epochs} epochs × "
          f"{iters_per_epoch} iters ({total_iters} total) "
          f"bs={cfg.batch_size} amp={use_amp} anom_prob={cfg.anomaly_prob}")
    print(f"      defect families : {fam_summary}")
    print(f"      memory_format   : channels_last (Tier 1)")
    log_every = max(1, n_epochs // 8)
    t0 = time.time()
    for epoch in range(n_epochs):
        net.train()
        loss_sum = 0.0
        n = 0
        for img, mask in loader:
            # Tier 1: move to device, then to channels_last layout. The
            # net weights are also channels_last (set in run_one_class),
            # so cudnn picks the channels_last conv kernels.
            img = img.to(device, non_blocking=True).to(
                    memory_format=torch.channels_last)
            mask = mask.to(device, non_blocking=True)
            # (v1.0 had torch.nan_to_num() here; dropped — the synthetic
            # pipeline doesn't produce NaNs and the op iterates the
            # whole tensor every step.)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = net(img)
                loss = focal_loss(logits, mask,
                                    gamma=cfg.focal_gamma,
                                    alpha=cfg.focal_alpha)

            # AMP grad-clip ordering: backward → unscale → clip → step → update
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            B = img.shape[0]
            loss_sum += loss.item() * B
            n += B
        if (epoch + 1) % log_every == 0 or epoch == n_epochs - 1:
            print(f"      epoch {epoch+1:>3}/{n_epochs}  "
                  f"loss={loss_sum/max(n,1):.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)
    net.eval()
    print(f"    [{now_hms()}] training done ({time.time()-t0:.1f}s)")


# ─────────────────────────────────────────────────────────────────────────────
# Inference primitives
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def _score_one_pass(net: SegUNet, x: torch.Tensor,
                     cfg: "RunConfig") -> torch.Tensor:
    """Per-pixel anomaly probability at the input resolution.
    Returns (B, H, W) on the configured device."""
    use_amp = (cfg.device.type == "cuda" and cfg.amp)
    with torch.amp.autocast("cuda", enabled=use_amp):
        logits = net(x)
    return F.softmax(logits.float(), dim=1)[:, 1]


@torch.inference_mode()
def score_batch(net: SegUNet, x: torch.Tensor,
                  cfg: "RunConfig") -> torch.Tensor:
    """TTA-averaged probability on CPU."""
    # Tier 1: convert input to channels_last so it matches the net's
    # memory layout. Flips don't preserve the layout perfectly, so we
    # re-apply contiguous(memory_format=...) on flipped variants.
    x = x.to(cfg.device, non_blocking=True).to(
            memory_format=torch.channels_last)
    acc = None; n = 0

    def _add(s):
        nonlocal acc, n
        if acc is None: acc = s.clone()
        else: acc += s
        n += 1

    _add(_score_one_pass(net, x, cfg))
    if cfg.tta in ("hflip", "hvflip"):
        xf = torch.flip(x, dims=[-1]).contiguous(
                memory_format=torch.channels_last)
        s = _score_one_pass(net, xf, cfg)
        _add(torch.flip(s, dims=[-1]))
    if cfg.tta in ("vflip", "hvflip"):
        xf = torch.flip(x, dims=[-2]).contiguous(
                memory_format=torch.channels_last)
        s = _score_one_pass(net, xf, cfg)
        _add(torch.flip(s, dims=[-2]))
    return (acc / max(n, 1)).cpu()


def _score_records(net, records, cfg, load_masks):
    ds = TextadInferenceDataset(records, input_size=cfg.input_size,
                                  load_masks=load_masks)
    loader = DataLoader(ds, batch_size=cfg.score_batch_size, shuffle=False,
                         **_build_loader_kwargs(cfg.num_workers))
    scores, gts = {}, {}
    n_done = 0; last_log = 0
    for x, masks, idxs in loader:
        sm = score_batch(net, x, cfg).numpy()
        m_np = masks.numpy()
        for b in range(sm.shape[0]):
            scores[int(idxs[b])] = sm[b]
            gts[int(idxs[b])] = m_np[b]
        n_done += sm.shape[0]
        if n_done - last_log >= 200:
            last_log = n_done
            print(f"      scored {n_done}/{len(records)}", flush=True)
    return scores, gts


# ─────────────────────────────────────────────────────────────────────────────
# Per-class pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_one_class(cls, records_all, cfg, run_dir, device,
                   taxonomy, local_saver=None) -> dict:
    hr(f"CLASS {cls}", "─")
    t_start = time.time()
    train_good = [r for r in records_all
                   if r.cls == cls and r.split == "train_good"]
    train_anom = [r for r in records_all
                   if r.cls == cls and r.split == "train_anomaly"]
    test       = [r for r in records_all
                   if r.cls == cls and r.split == "test"]
    print(f"  train_good={len(train_good)}  "
          f"train_anomaly={len(train_anom)}  test={len(test)}")
    if not train_good:
        return {"class": cls, "class_mean_ap": float("nan"),
                "eval_rows": [], "test_results": [], "elapsed_min": 0.0}

    specs = make_class_defect_specs(cls, taxonomy)
    print(f"  defect specs   : {[s.family for s in specs]}")

    net = SegUNet(in_channels=3, out_channels=2,
                    base=cfg.unet_base).to(device)
    # Tier 1: convert all conv weights to channels_last memory layout.
    # This needs to happen once per net (training + eval reuse it).
    net = net.to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  SegUNet: base={cfg.unet_base}  "
          f"params={n_params/1e6:.2f}M  "
          f"levels=4 (input must be %16==0)")

    train_textad(net, train_good, cfg, device, specs)

    if cfg.save_checkpoints:
        ck = run_dir / "ckpt" / f"{cls}_textad.pt"
        ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"net": net.state_dict(),
                    "defect_families": [s.family for s in specs]}, ck)
        print(f"    saved checkpoint -> {ck}")

    eval_rows = []
    class_mean_ap = float("nan")
    if not cfg.skip_eval and train_anom:
        sub(f"local validation  tta={cfg.tta}")
        scores, gts = _score_records(net, train_anom, cfg, load_masks=True)
        by_anom = defaultdict(list)
        for r_idx, sm in scores.items():
            r = train_anom[r_idx]
            sm_smooth = gaussian_smooth(sm, cfg.smooth_sigma)
            ap = pixel_average_precision(sm_smooth, gts[r_idx])
            by_anom[r.anomaly_type or "?"].append(ap)
            if local_saver is not None:
                # image_path REQUIRED for the v4 stacker's view parsing.
                local_saver.add(
                    cls=cls,
                    anomaly_type=r.anomaly_type or "unknown",
                    view_idx=int(r_idx),
                    score_map=sm_smooth,
                    gt_mask=gts[r_idx],
                    image_path=r.path)
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
        class_mean_ap = (float(np.mean(per_type_means))
                          if per_type_means else 0.0)
        print(f"    >>> class {cls} mean pixel-AP: {class_mean_ap:.4f}")

    test_results = []
    if not cfg.skip_submission and test:
        sub(f"scoring {len(test)} test images  tta={cfg.tta}")
        scores, _ = _score_records(net, test, cfg, load_masks=False)
        for r_idx, sm in scores.items():
            sm_smooth = gaussian_smooth(sm, cfg.smooth_sigma)
            sm_final = maybe_resize_to_submission(sm_smooth)
            test_results.append((test[r_idx], sm_final))

    elapsed_min = (time.time() - t_start) / 60.0
    print(f"  class {cls} done in {elapsed_min:.1f} min")
    del net
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {"class": cls, "class_mean_ap": class_mean_ap,
            "eval_rows": eval_rows, "test_results": test_results,
            "elapsed_min": elapsed_min}


def write_submission(all_test_results, run_dir: Path,
                      zip_it: bool = True) -> Path:
    sub("calibrating scores and writing submission.csv")
    scores = [sm for _, sm in all_test_results]
    if not scores: raise RuntimeError("no test scores")
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
# Run config and CLI
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RunConfig:
    data_root: Path
    report_dir: Path
    descriptions_csv: Path
    input_size: int = 256
    unet_base: int = 32
    # Training
    epochs: int = 200
    total_iters: int | None = 2500
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 0.0
    anomaly_prob: float = 0.7
    n_defects: tuple[int, int] = (1, 3)
    focal_gamma: float = 2.0
    focal_alpha: float = 0.5
    amp: bool = True
    num_workers: int = 8
    # Inference
    score_batch_size: int = 16
    smooth_sigma: float = 1.5
    tta: str = "hvflip"
    # Bookkeeping
    seed: int = 0
    only_classes: list[str] = field(default_factory=list)
    skip_eval: bool = False
    skip_submission: bool = False
    save_checkpoints: bool = False
    zip_submission: bool = True
    run_tag: str = ""
    device: torch.device | None = None


def make_run_id(cfg: RunConfig) -> str:
    fp = json.dumps({
        "method": "textad",
        "input_size": cfg.input_size,
        "unet_base": cfg.unet_base,
        "total_iters": cfg.total_iters,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "anomaly_prob": cfg.anomaly_prob,
        "n_defects": list(cfg.n_defects),
        "focal_gamma": cfg.focal_gamma,
        "focal_alpha": cfg.focal_alpha,
        "tta": cfg.tta,
        "smooth_sigma": cfg.smooth_sigma,
        "seed": cfg.seed,
        "v": "1.1",
    }, sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(fp).hexdigest()[:6]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    budget = (f"it{cfg.total_iters}"
              if (cfg.total_iters and cfg.total_iters > 0)
              else f"e{cfg.epochs}")
    bits = (f"{stamp}_textad_in{cfg.input_size}_b{cfg.unet_base}"
            f"_{budget}_bs{cfg.batch_size}"
            f"_nd{cfg.n_defects[0]}-{cfg.n_defects[1]}"
            f"_p{cfg.anomaly_prob:.2f}")
    if cfg.tta != "none":
        bits += f"_tta-{cfg.tta}"
    if cfg.run_tag:
        bits += f"_{re.sub(r'[^A-Za-z0-9._-]+', '-', cfg.run_tag)}"
    return f"{bits}_{digest}"


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--data-root",  type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    ap.add_argument("--descriptions-csv", type=Path,
                    default=DEFAULT_DESCRIPTIONS_CSV,
                    help="Per-class anomaly_descriptions.csv (the lantern). "
                         "Missing/unreadable file → fallback mixture for "
                         "every class.")
    ap.add_argument("--input-size", type=int, default=256,
                    help="Must be multiple of 16 (four 2x downsamples).")
    ap.add_argument("--unet-base", type=int, default=32,
                    help="Base channel count. 32 → ~7M params, "
                         "24 → ~4M, 48 → ~16M.")
    # Training
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--total-iters", type=int, default=2500,
                    help="Overrides --epochs.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--anomaly-prob", type=float, default=0.7,
                    help="Fraction of training samples that get a synthetic "
                         "defect. Higher than DRAEM's 0.5 because TextAD's "
                         "defects are more structured and class-appropriate.")
    ap.add_argument("--n-defects", type=int, nargs=2, default=[1, 3],
                    help="Min and max number of defects injected per "
                         "corrupted training image.")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--focal-alpha", type=float, default=0.5)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--num-workers", type=int, default=8)
    # Inference
    ap.add_argument("--score-batch-size", type=int, default=16)
    ap.add_argument("--smooth-sigma", type=float, default=1.5)
    ap.add_argument("--tta", default="hvflip",
                    choices=["none", "hflip", "vflip", "hvflip"])
    # Bookkeeping
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-classes", nargs="*", default=[])
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-submission", action="store_true")
    ap.add_argument("--save-checkpoints", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--no-save-local-preds", action="store_true",
                    help="Disable saving local_predictions.npz "
                         "(default: save). Disabling this breaks "
                         "downstream stacker training.")
    ap.add_argument("--run-tag", default="")
    args = ap.parse_args()

    if args.input_size % 16 != 0:
        raise SystemExit(f"[FATAL] --input-size must be multiple of 16 "
                          f"(SegUNet has 4 downsamples). "
                          f"Got {args.input_size}.")
    if args.n_defects[0] < 1 or args.n_defects[1] < args.n_defects[0]:
        raise SystemExit(f"[FATAL] --n-defects must be N_min <= N_max and "
                          f"N_min >= 1. Got {args.n_defects}.")

    cfg = RunConfig(
        data_root=args.data_root, report_dir=args.report_dir,
        descriptions_csv=args.descriptions_csv,
        input_size=args.input_size, unet_base=args.unet_base,
        epochs=args.epochs, total_iters=args.total_iters,
        batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay,
        anomaly_prob=args.anomaly_prob,
        n_defects=tuple(args.n_defects),
        focal_gamma=args.focal_gamma, focal_alpha=args.focal_alpha,
        amp=not args.no_amp, num_workers=args.num_workers,
        score_batch_size=args.score_batch_size,
        smooth_sigma=args.smooth_sigma, tta=args.tta,
        seed=args.seed, only_classes=args.only_classes,
        skip_eval=args.skip_eval, skip_submission=args.skip_submission,
        save_checkpoints=args.save_checkpoints,
        zip_submission=not args.no_zip,
        run_tag=args.run_tag,
    )
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # Tier 1 global flags. Cudnn picks the fastest conv kernel for our
    # fixed input shape; TF32 lets matmuls run faster on Ampere/Ada
    # without affecting AMP fp16 quality.
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = device

    run_id = make_run_id(cfg)
    run_dir = cfg.report_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with tee_to(run_dir / "run_log.txt"):
        hr(f"TEXTAD v1.1 — RUN {run_id}", "█")
        print(f"  data_root        : {cfg.data_root}")
        print(f"  run_dir          : {run_dir}")
        print(f"  descriptions_csv : {cfg.descriptions_csv}")
        print(f"  input_size       : {cfg.input_size}")
        print(f"  unet_base        : {cfg.unet_base}")
        if cfg.total_iters and cfg.total_iters > 0:
            print(f"  total_iters      : {cfg.total_iters} "
                  f"(overrides --epochs={cfg.epochs})")
        else:
            print(f"  epochs           : {cfg.epochs}")
        print(f"  batch_size       : {cfg.batch_size}")
        print(f"  lr / wd          : {cfg.lr} / {cfg.weight_decay}")
        print(f"  anomaly_prob     : {cfg.anomaly_prob}")
        print(f"  n_defects        : {cfg.n_defects[0]}–{cfg.n_defects[1]}")
        print(f"  focal gamma/alpha: {cfg.focal_gamma} / {cfg.focal_alpha}")
        print(f"  amp              : {cfg.amp}    "
              f"num_workers: {cfg.num_workers}")
        print(f"  score_batch_size : {cfg.score_batch_size}")
        print(f"  smooth_sigma     : {cfg.smooth_sigma}")
        print(f"  tta              : {cfg.tta}")
        print(f"  device           : {device}")
        print(f"  cudnn.benchmark  : True       "
              f"matmul precision : high (TF32)")
        if torch.cuda.is_available():
            print(f"                    {torch.cuda.get_device_name(0)}, "
                  f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

        with open(run_dir / "config.json", "w") as f:
            cfg_dump = {k: (str(v) if isinstance(v, (Path, torch.device))
                              else list(v) if isinstance(v, tuple) else v)
                          for k, v in asdict(cfg).items()}
            json.dump(cfg_dump, f, indent=2, default=str)

        # Lantern: load per-class taxonomy.
        sub("loading defect taxonomy from descriptions CSV")
        taxonomy = load_class_taxonomy(cfg.descriptions_csv)
        if taxonomy:
            for c in sorted(taxonomy):
                print(f"    {c:<12} → {taxonomy[c]}")
        else:
            print(f"    [warn] no taxonomy recovered from "
                  f"{cfg.descriptions_csv} — using fallback mixture for "
                  f"every class.")

        t_total = time.time()
        records = scan_dataset(cfg.data_root)
        if not records:
            print("\n[FATAL] no records found"); return
        classes = sorted({r.cls for r in records})
        if cfg.only_classes:
            classes = [c for c in classes if c in set(cfg.only_classes)]
        print(f"\n  running on {len(classes)} class(es): "
              f"{', '.join(classes)}")

        local_saver: LocalPredSaver | None = None
        if not cfg.skip_eval and not args.no_save_local_preds:
            local_saver = LocalPredSaver()

        all_test_results, all_eval_rows = [], []
        class_aps, class_elapsed = {}, {}
        for cls in classes:
            res = run_one_class(cls, records, cfg, run_dir, device,
                                  taxonomy=taxonomy,
                                  local_saver=local_saver)
            all_test_results.extend(res["test_results"])
            all_eval_rows.extend(res["eval_rows"])
            class_aps[cls] = res["class_mean_ap"]
            class_elapsed[cls] = res["elapsed_min"]

        if local_saver is not None and len(local_saver) > 0:
            local_saver.save(run_dir / "local_predictions.npz")

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
                w = csv.DictWriter(f,
                                     fieldnames=list(all_eval_rows[0].keys()))
                w.writeheader()
                for row in all_eval_rows: w.writerow(row)
            print(f"  saved per-(class, anomaly_type) AP -> {tab_path}")

        if not cfg.skip_submission and all_test_results:
            hr("SUBMISSION", "=")
            write_submission(all_test_results, run_dir,
                              zip_it=cfg.zip_submission)
            print(f"\n  Upload: {run_dir / 'submission.zip'}")

        master_csv = cfg.report_dir / "ablation_master.csv"
        row = {
            "run_id": run_id, "run_tag": cfg.run_tag,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backbone": "TEXTAD",
            "feature_layers": "",
            "target_layer": "",
            "input_size": cfg.input_size,
            "smooth_sigma": cfg.smooth_sigma, "tta": cfg.tta,
            "batch_size": cfg.batch_size,
            "score_batch_size": cfg.score_batch_size,
            "seed": cfg.seed, "n_classes": len(classes),
            **{f"AP_{c}": f"{class_aps.get(c, float('nan')):.4f}"
                for c in sorted(class_aps)},
            "AP_overall": f"{overall_ap:.4f}",
            "runtime_min": f"{(time.time() - t_total) / 60:.1f}",
            "submission_path": str(run_dir / "submission.zip")
                                if not cfg.skip_submission else "",
            "notes": (f"textad v1.1 base{cfg.unet_base} "
                      f"anom_prob={cfg.anomaly_prob} "
                      f"nd={cfg.n_defects[0]}-{cfg.n_defects[1]} "
                      f"{'it' + str(cfg.total_iters) if cfg.total_iters else 'e' + str(cfg.epochs)} "
                      f"bs{cfg.batch_size}"),
        }
        append_to_ablation_master(master_csv, row)
        print(f"\n  ablation row appended -> {master_csv}")
        hr(f"DONE — run_id={run_id}", "█")


if __name__ == "__main__":
    main()