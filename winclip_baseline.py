"""Spacepresso WinCLIP+ baseline — step 11 of the roadmap.

Implements "WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation"
(Jeong et al., CVPR 2023). The `+` suffix denotes the few-shot variant: a
small set of normal reference images augments the zero-shot text-image
alignment with a class-specific feature memory bank.

# Why WinCLIP+ is a good fusion partner at step 11

Existing tracks (PatchCore@WRN50 exp5, PatchCore@DINOv2 exp7, CutPaste-NN
exp8c, RD@WRN50 exp10b) all rely on visual-feature distance — either NN
over a coreset or a one-class reconstruction gap. They share the same
failure mode: small, semantically distinctive defects that look "OK" in
feature space but would be obvious to a human reading a defect catalog.

WinCLIP+ adds an orthogonal signal — text-image alignment — by using
the anomaly_descriptions.csv as a structured defect catalog. The
roadmap predicts the biggest gains on:

  - class_03 (gear)  — 3/4 anomaly types have rich descriptions, and
                        exp5 only reaches 0.17 mean AP here.
  - class_08 (capsule)— 6/7 have rich descriptions (cracks, layered caps,
                        fragments, indentations, stains, compressions).

# Architecture

  Pre-trained CLIP ViT-B/16 (OpenAI weights, loaded via open_clip). The
  visual transformer's patch tokens are projected through `ln_post` and
  `proj` so they live in the same 512-d text-aligned space as the text
  encoder's outputs — crucial for the cosine-similarity scoring to make
  sense.

  Multi-scale windowing (on the 14x14 patch grid):
    - Window 2x2 patches = 32x32 px regions, 13x13 starting positions.
    - Window 3x3 patches = 48x48 px regions, 12x12 starting positions.
    - Each window's feature = avg-pool over patches in the window,
      then L2-normalise.
    - Each window score is upsampled bilinearly to (224, 224) and the
      multi-scale maps are averaged.

  Zero-shot scoring per window w:
        logit_normal = cos(w, text_normal) * exp(logit_scale)
        logit_anom   = cos(w, text_anom)   * exp(logit_scale)
        p_anom       = softmax([logit_normal, logit_anom])[1]

  Few-shot scoring per test window w_test:
        score_fs = min over ref windows w_ref of (1 - cos(w_test, w_ref))
        (position-agnostic NN, robust to Spacepresso's multi-view drift)

  Combined:
        score = alpha * score_zs + (1 - alpha) * score_fs

# Text prompt construction (CPE + descriptions)

  Two sources are unioned:

  (a) Generic Compositional Prompt Ensemble (CPE):
        12 normal state words ("good", "flawless", "perfect", ...)
        × 10 templates ("a photo of a {state} {obj}", ...) = 120 prompts
        Same for anomalous side with state words like "damaged", "broken",
        "with defect", ..."

  (b) Description-grounded prompts from anomaly_descriptions.csv:
        For each (class, anomaly_type) row WHOSE description is rich
        (not the generic "Localized visual anomaly..." placeholder),
        we add:
          - "a photo of a defective {obj}: {description}"
          - Up to a handful of keyword-focused prompts derived from
            the description text (e.g. "a photo of a {obj} with a crack"
            when the description mentions "crack" or "fissured").

  All prompts are tokenised, encoded, mean-pooled per category, and
  L2-normalised before scoring.

# Memory and speed (ViT-B/16, 224 input, batch 32 on an NVIDIA L4)

  Forward pass:                 ~5 ms / image
  Text encoding (one-time/cls): ~2 sec
  Few-shot ref encoding (K=8):  ~0.1 sec
  Local eval (~40 imgs):        ~0.3 sec
  Test scoring (~740 imgs):     ~6 sec
  Per class total:              ~8 sec
  Full 8-class run:             ~1.5 min + model load ~30 sec

  Peak VRAM (ViT-B/16 fp16 AMP):  ~1.2 GB. Comfortable on 24 GB.

# Dependencies

  Adds: `open_clip_torch`. One-time install:

      uv pip install "open_clip_torch>=2.20"

  And one-time pre-flight (on a node with internet, e.g. login):

      python -c "import open_clip; \
          open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')"

  This caches the ~350 MB checkpoint under $HOME/.cache/clip so compute
  nodes can read it offline.

  Reuses (via import) from patchcore_baseline_v2.py:
    - ImageRecord, scan_dataset
    - pixel_average_precision, gaussian_smooth, calibrate_to_unit,
      float_matrix_to_q8rle, maybe_resize_to_submission
    - append_to_ablation_master
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
from torchvision import transforms

# Shared utilities — must live in same dir.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from patchcore_baseline_v2 import (
    ImageRecord,
    scan_dataset,
    pixel_average_precision,
    gaussian_smooth,
    calibrate_to_unit,
    float_matrix_to_q8rle,
    maybe_resize_to_submission,
    append_to_ablation_master,
)

try:
    import open_clip
    HAS_OPEN_CLIP = True
except Exception:
    HAS_OPEN_CLIP = False


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/work/u10813429/anomaly-detection")
DEFAULT_DATA_ROOT  = PROJECT_ROOT / "data"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "baseline_out"
DEFAULT_CSV        = PROJECT_ROOT / "data" / "anomaly_descriptions.csv"

# CLIP normalisation — DIFFERENT from ImageNet's. Using ImageNet stats
# silently degrades CLIP performance because the model was trained on
# this exact mean/std; the text-image alignment is calibrated to it.
CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

# CLIP ViT-B/16: patch size 16, 224x224 input -> 14x14 patch grid.
CLIP_INPUT_SIZE = 224
CLIP_PATCH_SIZE = 16
CLIP_GRID = CLIP_INPUT_SIZE // CLIP_PATCH_SIZE   # 14

# Default window sizes (in patches) — WinCLIP paper.
DEFAULT_WINDOW_SIZES = (2, 3)

# Generic-description marker. Rows whose description contains this phrase
# add no semantic signal beyond the state-word ensemble — we skip them.
GENERIC_PHRASE = "localized visual anomaly affecting the object surface"

# Object name display fallback — overridden by anomaly_descriptions.csv.
CLASS_FALLBACK_NAME = {
    "class_01": "resistor",
    "class_02": "inductor",
    "class_03": "gear",
    "class_04": "screw",
    "class_05": "nut",
    "class_06": "coffee bean",
    "class_07": "pistachio",
    "class_08": "capsule",
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
# Datasets — CLIP-normalised
# ─────────────────────────────────────────────────────────────────────────────
class CLIPImageDataset(Dataset):
    """Returns (image_tensor, mask_tensor_or_dummy, record_index)."""
    def __init__(self, records: list[ImageRecord], input_size: int,
                 load_masks: bool):
        self.records = records
        self.input_size = input_size
        self.load_masks = load_masks
        self.tx = transforms.Compose([
            transforms.Resize((input_size, input_size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
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


def worker_init_fn(_worker_id):
    base = torch.initial_seed() % 2 ** 32
    np.random.seed(base); random.seed(base)


# ─────────────────────────────────────────────────────────────────────────────
# Text prompts — Compositional Prompt Ensemble + description-grounded
# ─────────────────────────────────────────────────────────────────────────────
# State words. Lifted from the WinCLIP paper's appendix CPE table.
NORMAL_STATES = [
    "perfect", "flawless", "good", "normal", "unblemished",
    "pristine", "intact", "without any defect", "without any flaw",
    "clean", "undamaged", "with no anomaly",
]
ANOMALY_STATES = [
    "damaged", "defective", "with a defect", "with a flaw",
    "broken", "with damage", "with a visible defect",
    "anomalous", "with an imperfection", "imperfect",
    "abnormal", "with an anomaly",
]

# Templates that wrap a state word + object name.
TEMPLATES = [
    "a photo of a {state} {obj}",
    "a {state} {obj}",
    "a cropped photo of a {state} {obj}",
    "a close-up photo of a {state} {obj}",
    "a photo of one {state} {obj}",
    "a bright photo of a {state} {obj}",
    "this is a {state} {obj}",
    "a photo of the {state} {obj}",
    "a {state} {obj} in the image",
    "a manufacturing photo of a {state} {obj}",
]


# Keyword extractor used to lift focused prompts from rich descriptions.
# Each entry maps a list of trigger substrings to a short noun phrase that
# CLIP's text encoder handles well. Kept conservative — false positives
# pollute the anomaly text embedding and hurt zero-shot scoring.
KEYWORD_RULES = [
    (("broken", "fragment", "fracture"),                  "broken parts"),
    (("dent", "indent", "depress", "hollow", "dimpl"),    "a dent"),
    (("scratch",),                                         "a scratch"),
    (("groove",),                                          "a groove"),
    (("stain", "discolor", "blotchy", "darker patch"),     "a stain"),
    (("raised", "bulg", "protrus"),                        "a bulge"),
    (("crack", "fissur"),                                  "a crack"),
    (("mold", "fungal", "fuzzy", "powdery"),               "mold"),
    (("infest", "pest", "insect"),                         "pest damage"),
    (("contamination", "soil", "dirt", "oil residue"),     "contamination"),
    (("compress", "flatten"),                              "a flattened area"),
    (("layered cap", "additional cap", "extra cap"),       "an extra cap layer"),
    (("hole", "pitted"),                                   "holes"),
    (("rough edge", "jagged edge"),                        "rough edges"),
    (("rust",),                                            "rust"),
]


def extract_keywords(description: str) -> list[str]:
    """Return short noun-phrase keywords inferred from a rich description.
    Defensive: only triggers on substrings that have unambiguous defect
    semantics. Empty list if nothing matches (very rare in practice for
    Spacepresso's rich descriptions)."""
    out: list[str] = []
    desc_lower = description.lower()
    seen: set[str] = set()
    for triggers, phrase in KEYWORD_RULES:
        if phrase in seen:
            continue
        for t in triggers:
            if t in desc_lower:
                out.append(phrase)
                seen.add(phrase)
                break
    return out


def build_text_prompts(object_name: str,
                       per_type_descriptions: dict[str, str]
                       ) -> tuple[list[str], list[str]]:
    """Build (normal_prompts, anomaly_prompts) for one class.

    per_type_descriptions: {anomaly_type: description} from the CSV.
    """
    normal_prompts: list[str] = []
    anomaly_prompts: list[str] = []

    # (a) Always-included CPE.
    for state in NORMAL_STATES:
        for tmpl in TEMPLATES:
            normal_prompts.append(tmpl.format(state=state, obj=object_name))
    for state in ANOMALY_STATES:
        for tmpl in TEMPLATES:
            anomaly_prompts.append(tmpl.format(state=state, obj=object_name))

    # (b) Description-grounded prompts. Only rich descriptions add signal.
    # Dedup across (class, anomaly_type) since some rows are near-identical.
    added_descs: set[str] = set()
    added_keywords: set[str] = set()
    for _atype, desc in per_type_descriptions.items():
        if not desc:
            continue
        if GENERIC_PHRASE in desc.lower():
            continue
        d = desc.strip()
        if d not in added_descs:
            anomaly_prompts.append(f"a photo of a defective {object_name}: {d}")
            anomaly_prompts.append(f"a photo of an anomalous {object_name}: {d}")
            added_descs.add(d)
        for kw in extract_keywords(d):
            if kw in added_keywords:
                continue
            added_keywords.add(kw)
            # Keyword-focused prompts — short and clean, the kind CLIP
            # learned alignment for from the LAION/OpenAI text corpus.
            anomaly_prompts.append(f"a photo of a {object_name} with {kw}")
            anomaly_prompts.append(f"a {object_name} showing {kw}")
            anomaly_prompts.append(f"a close-up photo of {kw} on a {object_name}")

    return normal_prompts, anomaly_prompts


def load_descriptions_csv(csv_path: Path
                          ) -> dict[str, dict]:
    """Parse anomaly_descriptions.csv into:
        {class: {"object_name": str,
                 "per_type": {anomaly_type: description}}}.

    Robust to missing files — returns {} so the caller can fall back to
    pure-CPE prompts using CLASS_FALLBACK_NAME.
    """
    out: dict[str, dict] = {}
    if not csv_path.exists():
        print(f"  [warn] {csv_path} not found — using CPE-only prompts.")
        return out
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cls = row.get("public_class", "").strip()
            if not cls:
                continue
            obj = (row.get("object_name") or "").strip()
            atype = (row.get("public_anomaly") or "").strip()
            desc = (row.get("description") or "").strip()
            slot = out.setdefault(cls, {"object_name": obj, "per_type": {}})
            if not slot["object_name"]:
                slot["object_name"] = obj
            if atype:
                slot["per_type"][atype] = desc
    return out


# ─────────────────────────────────────────────────────────────────────────────
# WinCLIP+ predictor
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ClassFeatures:
    """Per-class state held during inference for a single class."""
    text_normal: torch.Tensor       # [D]
    text_anom: torch.Tensor         # [D]
    ref_windows: dict[int, torch.Tensor]  # {window_size: [K, H', W', D]}
    n_normal_prompts: int
    n_anom_prompts: int
    n_refs: int


class WinCLIPPredictor:
    """Thin wrapper around an open_clip ViT-B/16 model.

    The CLIP forward pass is re-implemented to expose patch tokens (and
    apply the same ln_post + proj as the model uses on the CLS token), so
    patch and text embeddings live in the same 512-d aligned space."""

    def __init__(self, model_name: str = "ViT-B-16",
                 pretrained: str = "openai",
                 device: torch.device | None = None,
                 amp: bool = True):
        if not HAS_OPEN_CLIP:
            raise SystemExit(
                "[FATAL] open_clip not installed. Install with:\n"
                "    uv pip install 'open_clip_torch>=2.20'")
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.amp = amp and self.device.type == "cuda"
        self.model_name = model_name
        self.pretrained = pretrained
        # Note: we replace open_clip's preprocess with our own Resize +
        # CLIP-normalize (so the pipeline matches what we feed every test
        # image in CLIPImageDataset).
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        # logit_scale: CLIP's temperature parameter, used in the zero-shot
        # softmax to match the calibration the model was trained with.
        self.logit_scale = self.model.logit_scale.detach().exp().item()
        # Sanity check the visual transformer geometry.
        v = self.model.visual
        self.embed_dim = v.conv1.out_channels
        self.patch_size = v.conv1.kernel_size[0]
        self.image_size = v.image_size if isinstance(v.image_size, int) \
            else v.image_size[0]
        if self.image_size != CLIP_INPUT_SIZE:
            print(f"  [warn] model image_size={self.image_size}; "
                  f"forcing input to {CLIP_INPUT_SIZE} via Resize.")
        if self.patch_size != CLIP_PATCH_SIZE:
            print(f"  [warn] model patch_size={self.patch_size}; "
                  f"this script assumes {CLIP_PATCH_SIZE}.")
        self.grid = CLIP_INPUT_SIZE // self.patch_size

    @torch.inference_mode()
    def encode_image_patches(self, x: torch.Tensor
                              ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cls_feat [B, D_out], patch_feat [B, P, D_out]) — both in
        CLIP's text-aligned space (after ln_post + proj)."""
        v = self.model.visual
        autocast = torch.amp.autocast("cuda", enabled=self.amp)
        with autocast:
            x = v.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            cls = v.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            x = torch.cat([cls, x], dim=1)
            x = x + v.positional_embedding.to(x.dtype)
            if hasattr(v, "patch_dropout") and v.patch_dropout is not None:
                x = v.patch_dropout(x)
            if hasattr(v, "ln_pre") and v.ln_pre is not None:
                x = v.ln_pre(x)
            x = v.transformer(x)
            x = v.ln_post(x)
            if getattr(v, "proj", None) is not None:
                x = x @ v.proj
        cls_feat = x[:, 0].float()
        patch_feat = x[:, 1:].float()
        return cls_feat, patch_feat

    @torch.inference_mode()
    def encode_text_ensemble(self, prompts: list[str],
                              batch_size: int = 128) -> torch.Tensor:
        """Encode prompts, L2-normalise each, mean-pool, L2-normalise the
        mean. Returns [D_out]."""
        accum = None
        n = 0
        for s in range(0, len(prompts), batch_size):
            chunk = prompts[s:s + batch_size]
            tokens = self.tokenizer(chunk).to(self.device)
            with torch.amp.autocast("cuda", enabled=self.amp):
                feats = self.model.encode_text(tokens)
            feats = feats.float()
            feats = F.normalize(feats, dim=-1)
            if accum is None: accum = feats.sum(dim=0)
            else:             accum += feats.sum(dim=0)
            n += feats.shape[0]
        mean = accum / max(n, 1)
        return F.normalize(mean, dim=-1)

    def window_features(self, patch_feat: torch.Tensor,
                         window_size: int) -> torch.Tensor:
        """patch_feat: [B, P, D] (P = grid*grid). Returns [B, H', W', D],
        L2-normalised per position. Window features = avg-pool with kernel
        `window_size` and stride 1 — same as WinCLIP."""
        B, P, D = patch_feat.shape
        H = W = int(math.isqrt(P))
        if H * W != P:
            raise ValueError(f"non-square patch grid: P={P}")
        pf = patch_feat.reshape(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W]
        pooled = F.avg_pool2d(pf, kernel_size=window_size, stride=1)
        pooled = pooled.permute(0, 2, 3, 1)  # [B, H', W', D]
        return F.normalize(pooled, dim=-1)

    @torch.inference_mode()
    def build_class_features(self,
                              normal_prompts: list[str],
                              anomaly_prompts: list[str],
                              ref_records: list[ImageRecord],
                              window_sizes: tuple[int, ...]
                              ) -> ClassFeatures:
        """One-time per-class setup: text embeddings + (optional) few-shot
        reference window features."""
        text_normal = self.encode_text_ensemble(normal_prompts)
        text_anom   = self.encode_text_ensemble(anomaly_prompts)

        ref_windows: dict[int, torch.Tensor] = {}
        if ref_records:
            ds = CLIPImageDataset(ref_records, input_size=CLIP_INPUT_SIZE,
                                    load_masks=False)
            loader = DataLoader(ds, batch_size=min(16, len(ds)),
                                 shuffle=False, num_workers=2,
                                 pin_memory=True)
            patch_list = []
            for x, _, _ in loader:
                x = x.to(self.device, non_blocking=True)
                _, pf = self.encode_image_patches(x)
                patch_list.append(pf.cpu())
            patches = torch.cat(patch_list, dim=0).to(self.device)
            for ws in window_sizes:
                ref_windows[ws] = self.window_features(patches, ws).contiguous()
            del patches

        return ClassFeatures(
            text_normal=text_normal,
            text_anom=text_anom,
            ref_windows=ref_windows,
            n_normal_prompts=len(normal_prompts),
            n_anom_prompts=len(anomaly_prompts),
            n_refs=len(ref_records),
        )

    def _zero_shot_score(self, window_feats: torch.Tensor,
                          text_normal: torch.Tensor,
                          text_anom: torch.Tensor) -> torch.Tensor:
        """window_feats: [B, H', W', D]. Returns [B, H', W'] anomaly prob."""
        B, H, W, D = window_feats.shape
        flat = window_feats.reshape(B * H * W, D)
        # Stack text features into [2, D] and compute logits in one matmul.
        text = torch.stack([text_normal, text_anom], dim=0)  # [2, D]
        logits = (flat @ text.T) * self.logit_scale          # [B*H*W, 2]
        prob = F.softmax(logits, dim=-1)
        return prob[..., 1].reshape(B, H, W)

    def _few_shot_score(self, test_feats: torch.Tensor,
                         ref_feats: torch.Tensor,
                         chunk: int = 4096) -> torch.Tensor:
        """test_feats: [B, H', W', D]. ref_feats: [K, H', W', D].

        Returns [B, H', W']: per-position (1 - max cosine sim to any
        reference window at any position). Position-agnostic NN matches
        WinCLIP+ and is robust to Spacepresso's multi-view drift."""
        B, H, W, D = test_feats.shape
        K, Hr, Wr, _ = ref_feats.shape
        flat_t = test_feats.reshape(B * H * W, D)
        flat_r = ref_feats.reshape(K * Hr * Wr, D)
        # cosine sim = matmul on already-L2-normalised features.
        out = torch.empty(B * H * W, device=test_feats.device)
        for s in range(0, flat_t.shape[0], chunk):
            e = min(flat_t.shape[0], s + chunk)
            sim = flat_t[s:e] @ flat_r.T            # [chunk, K*Hr*Wr]
            out[s:e] = 1.0 - sim.max(dim=-1).values
        return out.reshape(B, H, W)

    @torch.inference_mode()
    def _score_one_pass(self, x: torch.Tensor,
                         class_feats: ClassFeatures,
                         alpha: float,
                         window_sizes: tuple[int, ...]) -> torch.Tensor:
        """One forward pass per batch. Returns [B, 224, 224] score map."""
        x = x.to(self.device, non_blocking=True)
        _, patches = self.encode_image_patches(x)
        B = patches.shape[0]
        accum = torch.zeros(B, CLIP_INPUT_SIZE, CLIP_INPUT_SIZE,
                             device=self.device, dtype=torch.float32)
        for ws in window_sizes:
            wf = self.window_features(patches, ws)             # [B, H', W', D]
            zs = self._zero_shot_score(wf, class_feats.text_normal,
                                         class_feats.text_anom)  # [B, H', W']
            if class_feats.n_refs > 0 and ws in class_feats.ref_windows:
                fs = self._few_shot_score(wf, class_feats.ref_windows[ws])
                # zero-shot is in [0, 1]; few-shot is in [0, 2] but for
                # already-normalised vectors typically [0, 1] in practice.
                # We don't try to renormalise: a small ε bias is fine and
                # gets washed out by the global percentile calibration at
                # submission time.
                w_combined = alpha * zs + (1.0 - alpha) * fs
            else:
                w_combined = zs
            # Upsample window-grid score to image resolution.
            up = F.interpolate(w_combined.unsqueeze(1),
                               size=(CLIP_INPUT_SIZE, CLIP_INPUT_SIZE),
                               mode="bilinear", align_corners=False).squeeze(1)
            accum += up
        accum /= len(window_sizes)
        return accum.cpu()

    @torch.inference_mode()
    def score_batch(self, x: torch.Tensor,
                     class_feats: ClassFeatures,
                     alpha: float,
                     window_sizes: tuple[int, ...],
                     tta: str = "none") -> torch.Tensor:
        """TTA-aware wrapper. Averages anomaly maps over the requested
        flips/rotations after rotating each back to the canonical frame."""
        x = x.to(self.device, non_blocking=True)
        acc = None; n = 0

        def _add(s: torch.Tensor):
            nonlocal acc, n
            if acc is None: acc = s.clone()
            else: acc += s
            n += 1

        _add(self._score_one_pass(x, class_feats, alpha, window_sizes))
        if tta in ("hflip", "hvflip", "d4"):
            s = self._score_one_pass(torch.flip(x, dims=[-1]),
                                       class_feats, alpha, window_sizes)
            _add(torch.flip(s, dims=[-1]))
        if tta in ("vflip", "hvflip", "d4"):
            s = self._score_one_pass(torch.flip(x, dims=[-2]),
                                       class_feats, alpha, window_sizes)
            _add(torch.flip(s, dims=[-2]))
        if tta == "d4":
            for k in (1, 2, 3):
                s = self._score_one_pass(torch.rot90(x, k=k, dims=[-2, -1]),
                                           class_feats, alpha, window_sizes)
                _add(torch.rot90(s, k=-k, dims=[-2, -1]))
        return acc / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment tracking
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RunConfig:
    data_root: Path
    report_dir: Path
    csv: Path
    # Model
    clip_model: str = "ViT-B-16"
    clip_pretrained: str = "openai"
    amp: bool = True
    # Multi-scale windowing
    window_sizes: tuple[int, ...] = DEFAULT_WINDOW_SIZES
    # Scoring
    alpha: float = 0.5            # weight on zero-shot vs few-shot
    k_shot: int = 8               # number of few-shot reference images
    # Inference
    score_batch_size: int = 16
    num_workers: int = 4
    smooth_sigma: float = 1.5
    tta: str = "none"             # "none" | "hflip" | "vflip" | "hvflip" | "d4"
    # Bookkeeping
    seed: int = 0
    only_classes: list[str] = field(default_factory=list)
    skip_eval: bool = False
    skip_submission: bool = False
    zip_submission: bool = True
    save_prompts: bool = True
    run_tag: str = ""


def make_run_id(cfg: RunConfig) -> str:
    fp = json.dumps({
        "method": "winclip_plus",
        "clip_model": cfg.clip_model,
        "clip_pretrained": cfg.clip_pretrained,
        "window_sizes": list(cfg.window_sizes),
        "alpha": cfg.alpha,
        "k_shot": cfg.k_shot,
        "smooth_sigma": cfg.smooth_sigma,
        "tta": cfg.tta,
        "seed": cfg.seed,
        "v": 1,
    }, sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(fp).hexdigest()[:6]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    short = cfg.clip_model.lower().replace("-", "").replace("/", "")
    ws = "x".join(str(w) for w in cfg.window_sizes)
    bits = (f"{stamp}_winclip_{short}_w{ws}"
            f"_k{cfg.k_shot}_a{cfg.alpha:.2f}")
    if cfg.tta != "none":
        bits += f"_tta-{cfg.tta}"
    if cfg.run_tag:
        bits += f"_{re.sub(r'[^A-Za-z0-9._-]+', '-', cfg.run_tag)}"
    return f"{bits}_{digest}"


# ─────────────────────────────────────────────────────────────────────────────
# Per-class pipeline
# ─────────────────────────────────────────────────────────────────────────────
def select_reference_records(train_good: list[ImageRecord],
                              k: int, seed: int) -> list[ImageRecord]:
    """Pick K reference images for the few-shot bank. We avoid sampling
    multiple views of the same sample_id when possible — different views
    add real diversity, while duplicate views of the same physical sample
    add almost none. If K exceeds the number of unique sample_ids we fall
    back to filling with extra views."""
    if k <= 0 or not train_good:
        return []
    rng = random.Random(seed)
    by_sid: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in train_good:
        by_sid[r.sample_id or r.path.stem].append(r)
    sids = list(by_sid.keys())
    rng.shuffle(sids)
    picked: list[ImageRecord] = []
    # First pass: one view per sample_id.
    for sid in sids:
        if len(picked) >= k: break
        recs = by_sid[sid][:]
        rng.shuffle(recs)
        picked.append(recs[0])
    # Second pass: top-up with more views if needed.
    if len(picked) < k:
        remaining = [r for sid in sids for r in by_sid[sid][1:]]
        rng.shuffle(remaining)
        picked.extend(remaining[: k - len(picked)])
    return picked[:k]


def run_one_class(cls: str, records_all: list[ImageRecord],
                   predictor: WinCLIPPredictor,
                   descriptions: dict[str, dict],
                   cfg: RunConfig, run_dir: Path) -> dict:
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

    # ── Resolve object name + descriptions for this class
    info = descriptions.get(cls, {})
    object_name = (info.get("object_name") or CLASS_FALLBACK_NAME.get(cls)
                    or cls.replace("_", " "))
    per_type = info.get("per_type", {}) or {}
    n_rich = sum(1 for d in per_type.values()
                  if d and GENERIC_PHRASE not in d.lower())
    n_generic = len(per_type) - n_rich
    print(f"  object_name='{object_name}'  "
          f"descriptions: rich={n_rich} generic={n_generic}")

    # ── Build prompts and reference set
    normal_prompts, anomaly_prompts = build_text_prompts(object_name, per_type)
    print(f"  prompts: {len(normal_prompts)} normal, "
          f"{len(anomaly_prompts)} anomaly")
    if cfg.save_prompts:
        pdir = run_dir / "prompts"; pdir.mkdir(parents=True, exist_ok=True)
        with open(pdir / f"{cls}_normal.txt", "w") as f:
            f.write("\n".join(normal_prompts))
        with open(pdir / f"{cls}_anomaly.txt", "w") as f:
            f.write("\n".join(anomaly_prompts))
    ref_records = select_reference_records(train_good, cfg.k_shot, cfg.seed)
    if ref_records:
        sample_ids_in_refs = len({r.sample_id for r in ref_records})
        print(f"  few-shot: K={len(ref_records)} reference images "
              f"({sample_ids_in_refs} unique sample_ids)")

    # ── Encode text + references
    print(f"  [{now_hms()}] encoding text + reference patches...")
    class_feats = predictor.build_class_features(
        normal_prompts, anomaly_prompts, ref_records,
        window_sizes=cfg.window_sizes)

    # ── Local validation on train_anomaly (with masks)
    eval_rows: list[dict] = []
    class_mean_ap = float("nan")
    if not cfg.skip_eval and train_anom:
        sub(f"local validation — pixel-AP per (class, anomaly_type) "
            f"(tta={cfg.tta}, alpha={cfg.alpha}, K={cfg.k_shot})")
        ds_v = CLIPImageDataset(train_anom, input_size=CLIP_INPUT_SIZE,
                                  load_masks=True)
        loader_v = DataLoader(ds_v, batch_size=cfg.score_batch_size,
                               shuffle=False, num_workers=cfg.num_workers,
                               pin_memory=True,
                               persistent_workers=(cfg.num_workers > 0))
        scores_by_idx, gt_by_idx = {}, {}
        for x, masks, idxs in loader_v:
            sm = predictor.score_batch(x, class_feats, cfg.alpha,
                                         cfg.window_sizes, tta=cfg.tta).numpy()
            m_np = masks.numpy()
            for b in range(sm.shape[0]):
                s = gaussian_smooth(sm[b], cfg.smooth_sigma)
                scores_by_idx[int(idxs[b])] = s
                gt_by_idx[int(idxs[b])] = m_np[b]
        by_anom = defaultdict(list)
        for ridx, s in scores_by_idx.items():
            r = train_anom[ridx]
            by_anom[r.anomaly_type or "?"].append(
                pixel_average_precision(s, gt_by_idx[ridx]))
        print(f"    {'anomaly_type':<14} {'n_views':>8} "
              f"{'pixel-AP (mean ± std)':>26}")
        per_type_means: list[float] = []
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

    # ── Score the test set
    test_results: list[tuple[ImageRecord, np.ndarray]] = []
    if not cfg.skip_submission and test:
        sub(f"scoring {len(test)} test images "
            f"(tta={cfg.tta}, alpha={cfg.alpha}, K={cfg.k_shot})")
        ds_t = CLIPImageDataset(test, input_size=CLIP_INPUT_SIZE,
                                  load_masks=False)
        loader_t = DataLoader(ds_t, batch_size=cfg.score_batch_size,
                               shuffle=False, num_workers=cfg.num_workers,
                               pin_memory=True,
                               persistent_workers=(cfg.num_workers > 0))
        n_done, last_log = 0, 0
        for x, _, idxs in loader_t:
            sm = predictor.score_batch(x, class_feats, cfg.alpha,
                                         cfg.window_sizes, tta=cfg.tta).numpy()
            for b in range(sm.shape[0]):
                s = gaussian_smooth(sm[b], cfg.smooth_sigma)
                s = maybe_resize_to_submission(s)
                test_results.append((test[int(idxs[b])], s))
            n_done += sm.shape[0]
            if n_done - last_log >= 500:
                last_log = n_done
                print(f"      scored {n_done}/{len(test)}", flush=True)

    elapsed_min = (time.time() - t_start) / 60.0
    print(f"  class {cls} done in {elapsed_min:.1f} min")
    del class_feats
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
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--data-root",  type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    ap.add_argument("--csv",        type=Path, default=DEFAULT_CSV,
                    help="anomaly_descriptions.csv (description-grounded prompts)")
    # CLIP model
    ap.add_argument("--clip-model", default="ViT-B-16",
                    help="open_clip model name. ViT-B-16 (224 input, OpenAI) "
                         "is the WinCLIP default and is what this script is "
                         "tuned for. Larger backbones (ViT-L-14) work but "
                         "you should adjust --score-batch-size accordingly.")
    ap.add_argument("--clip-pretrained", default="openai",
                    help="open_clip pretrained tag (e.g. 'openai', "
                         "'laion2b_s34b_b88k').")
    ap.add_argument("--no-amp", action="store_true")
    # Windowing + scoring
    ap.add_argument("--window-sizes", type=int, nargs="+",
                    default=list(DEFAULT_WINDOW_SIZES),
                    help="Window sizes in patches (default: 2 3).")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Weight on zero-shot vs few-shot. 1.0 = pure WinCLIP "
                         "zero-shot; 0.0 = pure feature-NN over references.")
    ap.add_argument("--k-shot", type=int, default=8,
                    help="Number of normal references for the few-shot bank. "
                         "Use 0 for zero-shot ablation.")
    # Inference
    ap.add_argument("--score-batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--smooth-sigma", type=float, default=1.5)
    ap.add_argument("--tta", default="none",
                    choices=["none", "hflip", "vflip", "hvflip", "d4"])
    # Bookkeeping
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-classes", nargs="*", default=[])
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-submission", action="store_true")
    ap.add_argument("--no-save-prompts", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--run-tag", default="")
    args = ap.parse_args()

    cfg = RunConfig(
        data_root=args.data_root, report_dir=args.report_dir, csv=args.csv,
        clip_model=args.clip_model, clip_pretrained=args.clip_pretrained,
        amp=not args.no_amp,
        window_sizes=tuple(sorted(set(args.window_sizes))),
        alpha=args.alpha, k_shot=args.k_shot,
        score_batch_size=args.score_batch_size, num_workers=args.num_workers,
        smooth_sigma=args.smooth_sigma, tta=args.tta,
        seed=args.seed, only_classes=args.only_classes,
        skip_eval=args.skip_eval, skip_submission=args.skip_submission,
        zip_submission=not args.no_zip,
        save_prompts=not args.no_save_prompts,
        run_tag=args.run_tag,
    )
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    run_id = make_run_id(cfg)
    run_dir = cfg.report_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tee_to(run_dir / "run_log.txt"):
        hr(f"WINCLIP+ — RUN {run_id}", "█")
        print(f"  data_root        : {cfg.data_root}")
        print(f"  csv              : {cfg.csv}")
        print(f"  run_dir          : {run_dir}")
        print(f"  clip_model       : {cfg.clip_model}  "
              f"(pretrained={cfg.clip_pretrained})")
        print(f"  window_sizes     : {list(cfg.window_sizes)}  (patches)")
        print(f"  alpha            : {cfg.alpha}  "
              f"(zero-shot weight; few-shot = 1 - alpha)")
        print(f"  k_shot           : {cfg.k_shot}  "
              f"({'ZERO-SHOT only' if cfg.k_shot == 0 else 'few-shot reference bank'})")
        print(f"  score_batch_size : {cfg.score_batch_size}")
        print(f"  smooth_sigma     : {cfg.smooth_sigma}")
        print(f"  tta              : {cfg.tta}")
        print(f"  amp              : {cfg.amp}")
        print(f"  device           : {device}")
        if torch.cuda.is_available():
            print(f"                    {torch.cuda.get_device_name(0)}, "
                  f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        with open(run_dir / "config.json", "w") as f:
            json.dump({k: (list(v) if isinstance(v, tuple) else
                            str(v) if isinstance(v, Path) else v)
                       for k, v in asdict(cfg).items()}, f, indent=2)

        # Load CLIP once. All classes share it (only the text/ref features
        # change per class).
        print(f"\n  [{now_hms()}] loading CLIP {cfg.clip_model} "
              f"({cfg.clip_pretrained})...")
        predictor = WinCLIPPredictor(model_name=cfg.clip_model,
                                       pretrained=cfg.clip_pretrained,
                                       device=device, amp=cfg.amp)
        print(f"  CLIP loaded: embed_dim={predictor.embed_dim}, "
              f"patch_size={predictor.patch_size}, grid={predictor.grid}x{predictor.grid}, "
              f"logit_scale={predictor.logit_scale:.2f}")

        descriptions = load_descriptions_csv(cfg.csv)
        print(f"  loaded descriptions for {len(descriptions)} classes")

        t_total = time.time()
        records = scan_dataset(cfg.data_root)
        if not records:
            print("\n[FATAL] no records found"); return
        classes = sorted({r.cls for r in records})
        if cfg.only_classes:
            classes = [c for c in classes if c in set(cfg.only_classes)]
        print(f"\n  running on {len(classes)} class(es): {', '.join(classes)}")

        all_test_results = []
        all_eval_rows = []
        class_aps, class_elapsed = {}, {}
        for cls in classes:
            res = run_one_class(cls, records, predictor, descriptions,
                                  cfg, run_dir)
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
            print(f"  saved per-(class, anomaly_type) AP -> {tab_path}")

        if not cfg.skip_submission and all_test_results:
            hr("SUBMISSION", "=")
            write_submission(all_test_results, run_dir,
                              zip_it=cfg.zip_submission)
            print(f"\n  Upload: {run_dir / 'submission.zip'}")

        master_csv = cfg.report_dir / "ablation_master.csv"
        row = {
            "run_id": run_id,
            "run_tag": cfg.run_tag,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backbone": cfg.clip_model,
            "feature_layers": "",
            "target_layer": "",
            "input_size": CLIP_INPUT_SIZE,
            "smooth_sigma": cfg.smooth_sigma,
            "tta": cfg.tta,
            "batch_size": cfg.score_batch_size,
            "score_batch_size": cfg.score_batch_size,
            "seed": cfg.seed,
            "n_classes": len(classes),
            **{f"AP_{c}": f"{class_aps.get(c, float('nan')):.4f}"
                for c in sorted(class_aps)},
            "AP_overall": f"{overall_ap:.4f}",
            "runtime_min": f"{(time.time() - t_total) / 60:.1f}",
            "submission_path": str(run_dir / "submission.zip")
                                if not cfg.skip_submission else "",
            "notes": (f"winclip+ {cfg.clip_model} "
                      f"w={'+'.join(str(w) for w in cfg.window_sizes)} "
                      f"alpha={cfg.alpha} K={cfg.k_shot} tta={cfg.tta}"),
        }
        append_to_ablation_master(master_csv, row)
        print(f"\n  ablation row appended -> {master_csv}")
        hr(f"DONE — run_id={run_id}", "█")


if __name__ == "__main__":
    main()