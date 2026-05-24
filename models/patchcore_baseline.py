"""Spacepresso baseline: PatchCore + local pixel-AP harness + submission writer.

Course-grounded baseline (Roth et al., CVPR 2022 — exactly the method covered
in §4.6 of the Advanced Deep Learning notes). One script does the whole loop:

  1. SCAN dataset (already validated by analyze_spacepresso_dataset.py)
  2. TRAIN per-class PatchCore from class_XX/train/good
       — pre-trained WRN50 mid-level features (layers 2+3, the course's
         recommended layers — less ImageNet-class bias than the final block).
       — greedy k-center coreset subsampling on the memory bank.
  3. EVALUATE locally on class_XX/train/anomaly_*/ with masks from
     class_XX/ground_truth_train/anomaly_*/. Reports per-(class, anomaly_type)
     pixel-AP. Mean across classes ≈ your leaderboard estimate.
  4. SCORE the test set, Gaussian-smooth, q8rle-encode, write submission.csv.

The submission CSV is fully compliant with the challenge format:
    ID,Label
    img_xxxx_view1,q8rle 224 224 0 1 5 100 ...

Run (after `uv sync`):
    uv run python patchcore_baseline.py \\
        --data-root  /mnt/c/Users/Francoo/PycharmProjects/Anomaly-Detection/data \\
        --report-dir /mnt/c/Users/Francoo/PycharmProjects/Anomaly-Detection/baseline_out

Notes on memory:
  • Default config uses coreset 10% which fits in ~2 GB VRAM per class.
  • If you OOM on small GPUs, lower --coreset-frac to 0.05 or set
    --backbone resnet18 (smaller features, much faster, lower ceiling).
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
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
# Defaults — wired to the WSL project layout (override on the CLI)
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/mnt/c/Users/Francoo/PycharmProjects/Anomaly-Detection")
DEFAULT_DATA_ROOT  = PROJECT_ROOT / "data"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "baseline_out"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
INPUT_SIZE = 224  # all images in this dataset are 224x224 already

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

VIEW_RE = re.compile(r"^(?P<base>.+?)_view(?P<v>\d+)\.[A-Za-z]+$")


# ─────────────────────────────────────────────────────────────────────────────
# Tee logger so the run is preserved on disk
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
# Dataset record + scanner
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ImageRecord:
    path: Path
    cls: str
    split: str   # "train_good" | "train_anomaly" | "test"
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
        # train/good
        gd = cdir / "train" / "good"
        if gd.exists():
            for p in sorted(gd.iterdir()):
                if p.suffix.lower() in IMG_EXTS:
                    sid, v = parse_view(p.name)
                    out.append(ImageRecord(p, cls, "train_good",
                                           sample_id=sid, view=v))
        # train/anomaly_*
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
        # test (walks recursively, just in case)
        ted = cdir / "test"
        if ted.exists():
            for p in sorted(ted.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    sid, v = parse_view(p.name)
                    out.append(ImageRecord(p, cls, "test",
                                           sample_id=sid, view=v))
    return out


class SpacepressoDataset(Dataset):
    """Returns (image_tensor, mask_tensor_or_dummy, record_index)."""
    def __init__(self, records: list[ImageRecord], load_masks: bool = False):
        self.records = records
        self.load_masks = load_masks
        self.tx = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
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
                mm = mm.resize((INPUT_SIZE, INPUT_SIZE), Image.NEAREST)
                m = (np.asarray(mm) > 127).astype(np.float32)
        else:
            m = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        return x, torch.from_numpy(m), i


def make_loader(records, batch_size, load_masks=False, num_workers=2, shuffle=False):
    ds = SpacepressoDataset(records, load_masks=load_masks)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=(num_workers > 0))


# ─────────────────────────────────────────────────────────────────────────────
# Feature extractor — pre-trained CNN, mid-level layers (course §4.6)
# ─────────────────────────────────────────────────────────────────────────────
class FeatureExtractor(nn.Module):
    """Wraps a torchvision ResNet-family backbone and exposes layer2+layer3.

    Output of forward(x): (f2, f3) maps.
    Mid-level layers carry less ImageNet-class bias than the last block, per
    the PatchCore paper and the course notes.
    """

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
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.eval()
        for p in self.parameters(): p.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 3, 224, 224)  ->  (f2, f3)
        For WRN50 at 224 input: f2 ~ (B, 512, 28, 28), f3 ~ (B, 1024, 14, 14).
        """
        x = self.stem(x)
        x = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        return f2, f3


def patchify_and_combine(f2: torch.Tensor, f3: torch.Tensor,
                          patch_size: int = 3) -> torch.Tensor:
    """PatchCore-style patch features:
       1) average-pool each map with a (patch_size, patch_size) kernel so each
          location encodes a small neighbourhood;
       2) upsample f3 to f2's spatial resolution;
       3) concat along the channel dim;
       4) L2-normalise so the memory bank lives on a sphere.
    Returns (B, H*W, C_total) where (H, W) is f2's spatial size.
    """
    avg = nn.AvgPool2d(kernel_size=patch_size, stride=1, padding=patch_size // 2)
    f2 = avg(f2); f3 = avg(f3)
    f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
    feats = torch.cat([f2, f3], dim=1)
    B, C, H, W = feats.shape
    feats = feats.permute(0, 2, 3, 1).reshape(B, H * W, C)
    feats = F.normalize(feats, p=2, dim=-1)
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Greedy k-center coreset (Sener & Savarese 2018), as described in course §4.6
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def greedy_coreset(features: torch.Tensor, n_select: int,
                    seed: int = 0,
                    projection_dim: int | None = 32) -> torch.Tensor:
    """features: (N, D) on device. Returns int64 indices of the coreset.

    Random projection (projection_dim) speeds up cdist with only mild loss —
    matches the course's note about projections being used "only during
    core-set selection, without affecting the features stored in the bank."
    """
    device = features.device
    N, D = features.shape
    n_select = min(n_select, N)
    if projection_dim is not None and projection_dim < D:
        g = torch.Generator(device=device).manual_seed(seed)
        P = torch.randn(D, projection_dim, generator=g, device=device) / math.sqrt(projection_dim)
        feats_proj = features @ P
    else:
        feats_proj = features

    min_dist = torch.full((N,), float("inf"), device=device)
    rng = torch.Generator(device=device).manual_seed(seed + 1)
    first = int(torch.randint(0, N, (1,), generator=rng, device=device).item())
    selected = torch.empty(n_select, dtype=torch.long, device=device)
    selected[0] = first
    d0 = torch.cdist(feats_proj, feats_proj[first:first+1]).squeeze(1)
    min_dist = torch.minimum(min_dist, d0)

    log_every = max(1, n_select // 20)
    t0 = time.time()
    for i in range(1, n_select):
        idx = int(torch.argmax(min_dist).item())
        selected[i] = idx
        d_new = torch.cdist(feats_proj, feats_proj[idx:idx+1]).squeeze(1)
        min_dist = torch.minimum(min_dist, d_new)
        if i % log_every == 0 or i == n_select - 1:
            print(f"      coreset: {i+1}/{n_select} "
                  f"({(i+1)/n_select*100:>5.1f}%)  "
                  f"max-min-dist={min_dist.max().item():.4f}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# PatchCore class
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PatchCoreConfig:
    backbone: str = "wide_resnet50_2"
    coreset_frac: float = 0.10
    patch_size: int = 3
    knn_k: int = 9
    batch_size: int = 32
    num_workers: int = 2
    device: str = "cuda"
    seed: int = 0


class PatchCore:
    def __init__(self, cfg: PatchCoreConfig):
        self.cfg = cfg
        self.device = torch.device(
            cfg.device if torch.cuda.is_available() else "cpu")
        self.extractor = FeatureExtractor(cfg.backbone).to(self.device)
        self.memory: torch.Tensor | None = None
        self.feature_hw: tuple[int, int] | None = None
        self.feature_dim: int | None = None

    @torch.inference_mode()
    def _extract(self, loader: DataLoader) -> torch.Tensor:
        feats_list = []
        n = 0
        last_log = 0
        for x, _, _ in loader:
            batch_size = x.shape[0]
            x = x.to(self.device, non_blocking=True)

            f2, f3 = self.extractor(x)
            pf = patchify_and_combine(f2, f3, self.cfg.patch_size)

            if self.feature_hw is None:
                P = pf.shape[1]
                H = W = int(math.isqrt(P))
                self.feature_hw = (H, W)
                self.feature_dim = pf.shape[2]
            pf = pf.reshape(-1, pf.shape[-1]).detach().cpu()
            feats_list.append(pf)
            #feats_list.append(pf.reshape(-1, pf.shape[-1]).contiguous())

            del x, f2, f3, pf
            n += batch_size
            if n - last_log >= 256:
                last_log = n
                print(f"      extracted features from {n} images "
                      f"(feat dim={self.feature_dim}, "
                      f"patches/img={self.feature_hw[0] * self.feature_hw[1]})",
                      flush=True)
        return torch.cat(feats_list, dim=0)

    def fit(self, train_good_records: list[ImageRecord]) -> None:
        print(f"    [{now_hms()}] extracting train/good features "
              f"({len(train_good_records)} images)...")
        loader = make_loader(train_good_records,
                             batch_size=self.cfg.batch_size,
                             num_workers=self.cfg.num_workers,
                             load_masks=False, shuffle=False)
        all_feats = self._extract(loader)
        print(f"    -> {all_feats.shape[0]} patch features "
              f"({all_feats.element_size() * all_feats.numel() / 1e9:.2f} GB)")

        n_select = max(int(self.cfg.coreset_frac * all_feats.shape[0]), 1)
        print(f"    [{now_hms()}] greedy coreset: selecting {n_select} "
              f"of {all_feats.shape[0]} patches "
              f"({self.cfg.coreset_frac:.1%})")
        # idx = greedy_coreset(all_feats, n_select, seed=self.cfg.seed)
        # self.memory = all_feats[idx].contiguous()
        # del all_feats
        all_feats_gpu = all_feats.to(self.device, non_blocking=True)

        idx = greedy_coreset(
            all_feats_gpu,
            n_select,
            seed=self.cfg.seed
        )

        self.memory = all_feats_gpu[idx].contiguous()

        del all_feats

        torch.cuda.empty_cache()

        print(f"    [{now_hms()}] memory bank ready  "
              f"shape={tuple(self.memory.shape)}  "
              f"({self.memory.element_size() * self.memory.numel() / 1e6:.1f} MB)")

    @torch.inference_mode()
    def score_batch(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224). Returns (B, INPUT_SIZE, INPUT_SIZE) cpu tensor."""
        assert self.memory is not None, "fit() first"
        f2, f3 = self.extractor(x.to(self.device, non_blocking=True))
        pf = patchify_and_combine(f2, f3, self.cfg.patch_size)  # (B, P, C)
        B, P, C = pf.shape
        H = W = int(math.isqrt(P))
        flat = pf.reshape(-1, C)

        k = self.cfg.knn_k
        dist_min = torch.empty(flat.shape[0], device=self.device)
       # chunk = 4096
        chunk = 16384

        for s in range(0, flat.shape[0], chunk):
            q = flat[s:s + chunk]
            # d = torch.cdist(q, self.memory)
            # top = torch.topk(d, k=min(k, d.shape[1]), dim=1, largest=False).values
            # dist_min[s:s + chunk] = top[:, 0]   # nearest-neighbour distance
            sim = q @ self.memory.T
            dist = 1 - sim

            top = torch.topk(
                dist,
                k=min(k, dist.shape[1]),
                dim=1,
                largest=False
            ).values

            dist_min[s:s + chunk] = top[:, 0]

        score_map_lr = dist_min.reshape(B, H, W)
        score_map = F.interpolate(score_map_lr.unsqueeze(1),
                                  size=(INPUT_SIZE, INPUT_SIZE),
                                  mode="bilinear", align_corners=False)
        return score_map.squeeze(1).cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Pixel-level Average Precision (the competition metric)
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
# Score smoothing and global calibration
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
    """Robust min/max from the test-time score distribution: 1st / 99.5th pct.
    Used to map all scores into [0,1] before q8rle so the 8-bit quantisation
    covers the meaningful dynamic range. AP is rank-invariant so this does
    not change ordering — it only protects against tail outliers.
    """
    flat = np.concatenate([s.ravel() for s in scores])
    lo = float(np.percentile(flat, 1.0))
    hi = float(np.percentile(flat, 99.5))
    if hi <= lo: hi = lo + 1e-6
    return lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# q8rle (matches the competition specification exactly)
# ─────────────────────────────────────────────────────────────────────────────
def float_matrix_to_q8rle(x: np.ndarray) -> str:
    q = np.clip(np.rint(np.asarray(x, dtype=np.float32) * 255), 0, 255).astype(np.uint8)
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


def q8rle_to_float_matrix(s: str) -> np.ndarray:
    t = s.split()
    h, w = int(t[1]), int(t[2])
    vals = np.array(list(map(int, t[3::2])), dtype=np.uint8)
    lens = np.array(list(map(int, t[4::2])), dtype=np.int64)
    flat = np.repeat(vals, lens).reshape(w, h).T
    return flat.astype(np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# Per-class pipeline
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RunConfig:
    data_root: Path
    report_dir: Path
    backbone: str = "wide_resnet50_2"
    coreset_frac: float = 0.10
    batch_size: int = 32
    num_workers: int = 2
    smooth_sigma: float = 1.5
    knn_k: int = 9
    seed: int = 0
    only_classes: list[str] = field(default_factory=list)
    skip_eval: bool = False
    skip_submission: bool = False
    save_memory_banks: bool = True
    zip_submission: bool = True


def run_one_class(cls: str, records_all: list[ImageRecord],
                   cfg: RunConfig) -> dict:
    hr(f"CLASS {cls}", "─")
    t_start = time.time()

    train_good = [r for r in records_all if r.cls == cls and r.split == "train_good"]
    train_anom = [r for r in records_all if r.cls == cls and r.split == "train_anomaly"]
    test       = [r for r in records_all if r.cls == cls and r.split == "test"]
    print(f"  train_good={len(train_good)}  "
          f"train_anomaly={len(train_anom)}  test={len(test)}")

    pc_cfg = PatchCoreConfig(
        backbone=cfg.backbone, coreset_frac=cfg.coreset_frac,
        knn_k=cfg.knn_k, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, seed=cfg.seed,
    )
    pc = PatchCore(pc_cfg)
    pc.fit(train_good)

    if cfg.save_memory_banks:
        bank_path = cfg.report_dir / "banks" / f"{cls}_memory.pt"
        bank_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"memory": pc.memory.cpu(),
                    "feature_dim": pc.feature_dim,
                    "feature_hw": pc.feature_hw,
                    "config": pc_cfg.__dict__}, bank_path)
        print(f"    saved memory bank -> {bank_path}")

    # ── Local validation on train/anomaly_*
    eval_rows: list[dict] = []
    class_mean_ap = float("nan")
    if not cfg.skip_eval and train_anom:
        sub("local validation — per-anomaly-type pixel-AP")
        loader = make_loader(train_anom, batch_size=cfg.batch_size,
                             num_workers=cfg.num_workers, load_masks=True)
        scores_per_idx: dict[int, np.ndarray] = {}
        gt_per_idx: dict[int, np.ndarray] = {}
        with torch.inference_mode():
            for x, masks, idxs in loader:
                score_maps = pc.score_batch(x).numpy()
                masks_np = masks.numpy()
                for b in range(x.shape[0]):
                    sm = gaussian_smooth(score_maps[b], cfg.smooth_sigma)
                    scores_per_idx[int(idxs[b])] = sm
                    gt_per_idx[int(idxs[b])] = masks_np[b]

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

    # ── Score the test set
    test_results: list[tuple[ImageRecord, np.ndarray]] = []
    if not cfg.skip_submission and test:
        sub(f"scoring {len(test)} test images")
        loader = make_loader(test, batch_size=cfg.batch_size,
                             num_workers=cfg.num_workers, load_masks=False)
        n_done = 0
        last_log = 0
        with torch.inference_mode():
            for x, _, idxs in loader:
                score_maps = pc.score_batch(x).numpy()
                for b in range(x.shape[0]):
                    sm = gaussian_smooth(score_maps[b], cfg.smooth_sigma)
                    test_results.append((test[int(idxs[b])], sm))
                n_done += x.shape[0]
                if n_done - last_log >= 200:
                    last_log = n_done
                    print(f"      scored {n_done}/{len(test)}", flush=True)

    elapsed = time.time() - t_start
    print(f"  class {cls} done in {elapsed/60:.1f} min")

    del pc
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return {
        "class": cls,
        "class_mean_ap": class_mean_ap,
        "eval_rows": eval_rows,
        "test_results": test_results,
    }


def write_submission(all_test_results: list[tuple[ImageRecord, np.ndarray]],
                     report_dir: Path, zip_it: bool = True) -> Path:
    sub("calibrating scores and writing submission.csv")
    scores = [sm for _, sm in all_test_results]
    if not scores:
        raise RuntimeError("no test scores to write")
    lo, hi = calibrate_to_unit(scores)
    print(f"    global score calibration  lo={lo:.4f}  hi={hi:.4f}")

    csv_path = report_dir / "submission.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Label"])
        for r, sm in all_test_results:
            normed = np.clip((sm - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
            row_id = r.path.stem
            w.writerow([row_id, float_matrix_to_q8rle(normed)])
            n += 1
    print(f"    wrote {n} rows -> {csv_path}")

    if zip_it:
        zip_path = csv_path.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv_path, arcname=csv_path.name)
        print(f"    zipped         -> {zip_path}")
        return zip_path
    return csv_path


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--data-root",  type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    ap.add_argument("--backbone", default="wide_resnet50_2",
                    choices=["wide_resnet50_2", "resnet50", "resnet18"])
    ap.add_argument("--coreset-frac", type=float, default=0.10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--smooth-sigma", type=float, default=1.5)
    ap.add_argument("--knn-k", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-classes", nargs="*", default=[],
                    help="limit to a subset, e.g. --only-classes class_01 class_06")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-submission", action="store_true")
    ap.add_argument("--no-save-banks", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    cfg = RunConfig(
        data_root=args.data_root, report_dir=args.report_dir,
        backbone=args.backbone, coreset_frac=args.coreset_frac,
        batch_size=args.batch_size, num_workers=args.num_workers,
        smooth_sigma=args.smooth_sigma, knn_k=args.knn_k, seed=args.seed,
        only_classes=args.only_classes, skip_eval=args.skip_eval,
        skip_submission=args.skip_submission,
        save_memory_banks=not args.no_save_banks,
        zip_submission=not args.no_zip,
    )
    cfg.report_dir.mkdir(parents=True, exist_ok=True)

    with tee_to(cfg.report_dir / "run_log.txt"):
        hr("PATCHCORE BASELINE — SPACEPRESSO", "█")
        print(f"  data_root  : {cfg.data_root}")
        print(f"  report_dir : {cfg.report_dir}")
        print(f"  backbone   : {cfg.backbone}")
        print(f"  coreset    : {cfg.coreset_frac:.1%}   knn_k={cfg.knn_k}")
        print(f"  smoothing  : sigma={cfg.smooth_sigma}")
        print(f"  device     : {'cuda' if torch.cuda.is_available() else 'cpu'}")
        if torch.cuda.is_available():
            print(f"               {torch.cuda.get_device_name(0)}, "
                  f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

        records = scan_dataset(cfg.data_root)
        if not records:
            print("\n[FATAL] no records found — abort"); return
        classes = sorted({r.cls for r in records})
        if cfg.only_classes:
            classes = [c for c in classes if c in set(cfg.only_classes)]
        print(f"\n  running on {len(classes)} class(es): {', '.join(classes)}")

        all_test_results: list[tuple[ImageRecord, np.ndarray]] = []
        all_eval_rows: list[dict] = []
        class_aps: dict[str, float] = {}
        for cls in classes:
            res = run_one_class(cls, records, cfg)
            all_test_results.extend(res["test_results"])
            all_eval_rows.extend(res["eval_rows"])
            class_aps[cls] = res["class_mean_ap"]

        hr("LOCAL VALIDATION SUMMARY", "=")
        print(f"  {'class':<10} {'mean pixel-AP':>15}")
        for cls in classes:
            print(f"  {cls:<10} {class_aps.get(cls, float('nan')):>15.4f}")
        valid_aps = [v for v in class_aps.values() if not math.isnan(v)]
        if valid_aps:
            mean_ap = float(np.mean(valid_aps))
            print(f"  {'OVERALL':<10} {mean_ap:>15.4f}")
            print(f"\n  (your local proxy for the leaderboard public AP — "
                  f"expect a 0.05-0.15 gap on the test set, in either direction.)")

        if all_eval_rows:
            tab_path = cfg.report_dir / "local_eval.csv"
            with open(tab_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(all_eval_rows[0].keys()))
                w.writeheader()
                for row in all_eval_rows: w.writerow(row)
            print(f"  saved per-(class, anomaly_type) AP table -> {tab_path}")

        if not cfg.skip_submission and all_test_results:
            hr("SUBMISSION", "=")
            out_path = write_submission(all_test_results, cfg.report_dir,
                                         zip_it=cfg.zip_submission)
            print(f"\n  Upload to the Kaggle leaderboard:\n    {out_path}")

        hr(f"DONE — see {cfg.report_dir}", "█")


if __name__ == "__main__":
    main()