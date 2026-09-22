"""Class-aware procedural defect synthesis.

Ported from ``exploit_dataset/defect_library.py``, which lived in the
dataset-analysis directory even though TextAD depended on it at training time.

Eight defect families, each a ``(shape, appearance)`` pair: the shape function
draws a binary mask, the appearance function paints that region into the
image. They are always produced together — a synthesis routine that can emit
an image without its matching mask is a training-label bug waiting to happen.

  scratch        thin straight line, 1-4 px, dark or bright
  crack          jagged random-walk line, 1-2 px, dark
  dent           smooth ellipse, darkened with a soft shadow
  bulge          smooth ellipse, brightened like a specular highlight
  stain          irregular blob blended with a dark colour
  fragment       large irregular region replaced with a background tint
  mold           cluster of small overlapping greenish blobs
  contamination  scattered small dark dots, pest-damage style

Which families apply to which class is read from
``data/anomaly_descriptions.csv`` by keyword. Without the CSV every class
falls back to a sensible mixture, so training still works — the descriptions
sharpen the synthesis rather than enabling it.

Every generator takes an explicit ``numpy.random.Generator``, so augmentation
is reproducible from the run seed and still differs across workers.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from spacepresso.core.logging import get_logger

__all__ = [
    "APPEARANCE_GENERATORS",
    "FAMILIES",
    "SHAPE_GENERATORS",
    "DefectSpec",
    "families_from_description",
    "inject_defects",
    "load_class_taxonomy",
    "make_class_defect_specs",
]

logger = get_logger(__name__)

_KEYWORD_TO_FAMILY: list[tuple[tuple[str, ...], str]] = [
    (
        (
            "scratch",
            "linear or",
            "linear mark",
            "groove",
            "shallow groove",
            "linear or jagged",
        ),
        "scratch",
    ),
    (("crack", "fissure", "fissured", "linear or jagged line", "jagged line"), "crack"),
    (
        (
            "dent",
            "depressed",
            "indent",
            "dimpled",
            "hollow",
            "concave",
            "flattened",
            "compression",
        ),
        "dent",
    ),
    (("raised", "bulging", "protrusion", "layered cap", "extra layer"), "bulge"),
    (
        (
            "stain",
            "blotchy",
            "discolor",
            "patchy",
            "darker patch",
            "darker shade",
            "soiling",
            "soiled",
            "irregular or blotchy",
            "dark or discolored patches",
        ),
        "stain",
    ),
    (
        ("fuzzy", "powdery", "mold", "fungal", "moisture exposure", "fungal growth"),
        "mold",
    ),
    (
        (
            "infestation",
            "pest",
            "pitted",
            "holes",
            "small spot",
            "fleck",
            "round or irregular",
            "irregular holes or discolorations",
        ),
        "contamination",
    ),
    (
        (
            "fragment",
            "broken",
            "irregular fragments",
            "fragments or pieces",
            "jagged fragments",
            "rough edges",
            "irregular or jagged",
            "structural failure",
        ),
        "fragment",
    ),
]

# Default mixture when the CSV is missing or the description is generic
# ("Localized visual anomaly affecting the object surface.").
_FALLBACK_FAMILIES = ("scratch", "stain", "dent", "fragment", "bulge")


def families_from_description(desc: str) -> list[str]:
    """Return the ordered, de-duplicated list of defect families
    matched by the description string. Empty list means no keyword
    fired — caller should apply the fallback mixture."""
    d = desc.lower()
    out: list[str] = []
    for keywords, family in _KEYWORD_TO_FAMILY:
        if family not in out and any(k in d for k in keywords):
            out.append(family)
    return out


def load_class_taxonomy(
    csv_path: Path | None, fallback: tuple[str, ...] = _FALLBACK_FAMILIES
) -> dict[str, list[str]]:
    """Returns {class_name: [defect families]} ordered by descending
    count of CSV rows matching that family.

    The CSV is expected to have columns at least:
      public_class       (e.g. 'class_01')
      description        (free text; may be generic for some rows)

    Missing/unreadable CSV → empty dict; the model code then applies
    the fallback mixture to every class."""
    out: dict[str, list[str]] = {}
    if csv_path is None or not csv_path.exists():
        return out
    raw: dict[str, list[str]] = {}
    try:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cls = row.get("public_class", "").strip()
                desc = row.get("description", "")
                if not cls:
                    continue
                fams = families_from_description(desc)
                if not fams:
                    fams = list(fallback)
                raw.setdefault(cls, []).extend(fams)
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        logger.warning(
            "could not parse %s: %s — using the fallback mixture", csv_path, exc
        )
        return out
    for cls, fams in raw.items():
        c = Counter(fams)
        out[cls] = [f for f, _ in c.most_common()]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 perf helpers
# ─────────────────────────────────────────────────────────────────────────────
_INDICES_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _get_indices(H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """Module-cached np.indices((H, W)). Returns (ys, xs). Allocated once
    per (H, W) per worker process; subsequent calls are O(1)."""
    key = (H, W)
    cached = _INDICES_CACHE.get(key)
    if cached is None:
        cached = tuple(np.indices((H, W)))  # (ys, xs)
        _INDICES_CACHE[key] = cached
    return cached


def _gauss(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur, preferring OpenCV and falling back to SciPy.

    OpenCV is 5-10x faster on the float32 maps this module produces, and the
    defect synthesis runs inside the dataloader where that matters. The two
    agree to sub-pixel differences at the boundary (BORDER_REFLECT against
    SciPy's mode="reflect"), which has no effect on the masks.

    The original imported cv2 at module scope and crashed with a NameError
    when it was absent, despite documenting SciPy as the reference behaviour.
    """
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    kernel = max(3, 2 * int(round(3.0 * sigma)) + 1)

    try:
        import cv2
    except ImportError:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(arr, sigma=sigma, mode="reflect").astype(np.float32)

    return cv2.GaussianBlur(
        arr,
        (kernel, kernel),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT,
    )


def _percentile_fast(arr: np.ndarray, pct: float) -> float:
    """Equivalent of np.percentile(arr, pct) computed via np.partition.
    O(n) instead of O(n log n) — about 5× faster on 256×256 inputs.

    For pct=p, we want the value v such that p% of entries are ≤ v.
    That's the (n - k)-th order statistic where k = round(n * (100-p)/100)."""
    n = arr.size
    k = max(1, min(n - 1, int(round(n * (100.0 - pct) / 100.0))))
    return float(np.partition(arr.ravel(), n - k)[n - k])


# ─────────────────────────────────────────────────────────────────────────────
# Shape generators — each returns a binary (H, W) float32 mask in {0, 1}
# ─────────────────────────────────────────────────────────────────────────────
def _empty_mask(H: int, W: int) -> np.ndarray:
    return np.zeros((H, W), dtype=np.float32)


def shape_scratch(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """Thin straight line, width 1–4 px, random orientation."""
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    x0 = int(rng.integers(0, W))
    y0 = int(rng.integers(0, H))
    length = int(rng.integers(max(8, min(H, W) // 5), max(16, min(H, W) // 2)))
    angle = float(rng.uniform(0, 2 * np.pi))
    x1 = int(np.clip(x0 + length * np.cos(angle), 0, W - 1))
    y1 = int(np.clip(y0 + length * np.sin(angle), 0, H - 1))
    width = int(rng.integers(1, 4))
    draw.line([(x0, y0), (x1, y1)], fill=255, width=width)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return (arr > 0.5).astype(np.float32)


def shape_crack(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """Jagged thin line via 2-D random walk. Width 1–2 px."""
    n_steps = int(rng.integers(min(H, W) // 4, max(min(H, W) // 4 + 1, min(H, W) // 2)))
    x = float(rng.integers(0, W))
    y = float(rng.integers(0, H))
    angle = float(rng.uniform(0, 2 * np.pi))
    pts = [(x, y)]
    for _ in range(n_steps):
        angle += float(rng.normal(0, 0.3))
        x += float(np.cos(angle) * 1.5)
        y += float(np.sin(angle) * 1.5)
        pts.append((max(0, min(W - 1, x)), max(0, min(H - 1, y))))
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    width = int(rng.integers(1, 3))
    for i in range(len(pts) - 1):
        draw.line([pts[i], pts[i + 1]], fill=255, width=width)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return (arr > 0.5).astype(np.float32)


def shape_dent(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth ellipse anywhere in the central 50% of the image."""
    cx = int(rng.integers(W // 4, 3 * W // 4))
    cy = int(rng.integers(H // 4, 3 * H // 4))
    rx = int(rng.integers(max(3, min(H, W) // 24), max(6, min(H, W) // 8)))
    ry = int(rng.integers(max(3, min(H, W) // 24), max(6, min(H, W) // 8)))
    angle = float(rng.uniform(0, np.pi))
    ys, xs = _get_indices(H, W)
    dx = xs - cx
    dy = ys - cy
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    xr = dx * cos_a + dy * sin_a
    yr = -dx * sin_a + dy * cos_a
    d = (xr / max(rx, 1)) ** 2 + (yr / max(ry, 1)) ** 2
    return (d <= 1.0).astype(np.float32)


def shape_bulge(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """Mask geometry identical to dent; difference is in appearance."""
    return shape_dent(H, W, rng)


def shape_stain(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """Irregular blob via Gaussian-smoothed random noise + threshold."""
    noise = rng.random((H, W), dtype=np.float32)
    sigma = float(rng.uniform(5.0, 12.0))
    smooth = _gauss(noise, sigma)
    pct = float(rng.uniform(80, 92))
    th = _percentile_fast(smooth, pct)
    return (smooth > th).astype(np.float32)


def shape_fragment(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """Bigger irregular region, sometimes biased toward an image edge
    (broken edges/corners are a common defect mode in this dataset)."""
    noise = rng.random((H, W), dtype=np.float32)
    sigma = float(rng.uniform(8.0, 18.0))
    smooth = _gauss(noise, sigma)
    pct = float(rng.uniform(85, 95))
    th = _percentile_fast(smooth, pct)
    mask = (smooth > th).astype(np.float32)
    if rng.random() < 0.4:
        # In-place edge bias: zero out the half of the mask away from
        # the chosen image edge. Equivalent to the v1.0 mask*edge dance,
        # but saves the (H, W) edge allocation + a multiply.
        side = int(rng.integers(0, 4))
        margin = max(8, int(min(H, W) * 0.3))
        if side == 0:
            mask[margin:, :] = 0.0  # keep top
        elif side == 1:
            mask[:-margin, :] = 0.0  # keep bottom
        elif side == 2:
            mask[:, margin:] = 0.0  # keep left
        else:
            mask[:, :-margin] = 0.0  # keep right
    return mask


def _draw_disks(
    height: int,
    width: int,
    centres: list[tuple[int, int]],
    radii: list[int],
) -> np.ndarray:
    """Filled disks on a binary canvas, via PIL.

    The original called ``cv2.circle`` here but only imported cv2 inside one
    other function, so these two shape generators raised ``NameError`` the
    moment they were selected — which, for a class whose taxonomy included
    "mold" or "contamination", meant the defect family silently never fired.
    PIL is already a hard dependency; OpenCV was not.

    Out-of-bounds centres are clipped by the draw, as cv2 did.
    """
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for (cx, cy), radius in zip(centres, radii, strict=True):
        draw.ellipse([(cx - radius, cy - radius), (cx + radius, cy + radius)], fill=1)
    return np.asarray(canvas, dtype=np.float32)


def shape_mold(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """A cluster of 4-12 small overlapping disks — a fuzzy fungal patch."""
    n_blobs = int(rng.integers(4, 13))
    cx_base = int(rng.integers(W // 4, 3 * W // 4))
    cy_base = int(rng.integers(H // 4, 3 * H // 4))
    spread = max(8, min(H, W) // 10)
    centres = [
        (
            cx_base + int(rng.integers(-spread, spread)),
            cy_base + int(rng.integers(-spread, spread)),
        )
        for _ in range(n_blobs)
    ]
    radii = [int(rng.integers(2, 8)) for _ in range(n_blobs)]
    return _draw_disks(H, W, centres, radii)


def shape_contamination(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """Scattered small dark dots, in the style of pest damage."""
    n_dots = int(rng.integers(8, 30))
    cx_base = int(rng.integers(W // 4, 3 * W // 4))
    cy_base = int(rng.integers(H // 4, 3 * H // 4))
    spread = max(12, min(H, W) // 6)
    centres = [
        (
            cx_base + int(rng.integers(-spread, spread)),
            cy_base + int(rng.integers(-spread, spread)),
        )
        for _ in range(n_dots)
    ]
    radii = [int(rng.integers(1, 4)) for _ in range(n_dots)]
    return _draw_disks(H, W, centres, radii)


SHAPE_GENERATORS: dict[str, Callable] = {
    "scratch": shape_scratch,
    "crack": shape_crack,
    "dent": shape_dent,
    "bulge": shape_bulge,
    "stain": shape_stain,
    "fragment": shape_fragment,
    "mold": shape_mold,
    "contamination": shape_contamination,
}


# ─────────────────────────────────────────────────────────────────────────────
# Appearance generators — each takes (image01, mask, rng) and returns the
# corrupted image (mask is left untouched; "image and mask grow as twins").
# ─────────────────────────────────────────────────────────────────────────────
def appearance_scratch(
    img: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    bright = bool(rng.random() < 0.5)
    base = np.array([0.95, 0.95, 0.95]) if bright else np.array([0.05, 0.05, 0.05])
    color = np.clip(base + rng.normal(0, 0.05, 3), 0, 1).astype(np.float32)
    m3 = mask.astype(np.float32)[..., None]
    return ((1.0 - m3) * img + m3 * color[None, None, :]).astype(np.float32)


def appearance_crack(
    img: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Dark thin line + slight darkening of immediate neighborhood."""
    soft = _gauss(mask, 1.5)
    color = np.clip(np.array([0.05, 0.05, 0.05]) + rng.normal(0, 0.03, 3), 0, 1).astype(
        np.float32
    )
    m3 = soft[..., None]
    return ((1.0 - m3) * img + m3 * color[None, None, :]).astype(np.float32)


def appearance_dent(
    img: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Multiplicative darkening with soft edge fade."""
    soft = _gauss(mask, 2.0)
    darken = float(rng.uniform(0.40, 0.70))
    m3 = soft[..., None]
    return np.clip(img * (1.0 - darken * m3), 0, 1).astype(np.float32)


def appearance_bulge(
    img: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Lift toward white (specular-highlight feel) with soft edge."""
    soft = _gauss(mask, 2.0)
    brighten = float(rng.uniform(0.30, 0.60))
    m3 = soft[..., None]
    return np.clip(img + brighten * m3 * (1.0 - img), 0, 1).astype(np.float32)


def appearance_stain(
    img: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Alpha-blend with a stain colour (mostly dark/brown/red)."""
    palette = [
        np.array([0.20, 0.10, 0.05]),  # dark brown (oil/rust)
        np.array([0.30, 0.10, 0.10]),  # dark red
        np.array([0.40, 0.40, 0.40]),  # gray
        np.array([0.15, 0.20, 0.10]),  # dark green
        np.array([0.10, 0.10, 0.10]),  # near black
    ]
    base = palette[int(rng.integers(0, len(palette)))]
    color = np.clip(base + rng.normal(0, 0.05, 3), 0, 1).astype(np.float32)
    alpha = float(rng.uniform(0.50, 0.90))
    m3 = mask.astype(np.float32)[..., None]
    return ((1.0 - alpha * m3) * img + (alpha * m3) * color[None, None, :]).astype(
        np.float32
    )


def appearance_fragment(
    img: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Replace the masked region with a tinted background colour + noise
    — simulates a broken-away piece.

    v1.0 took the mean of pixels OUTSIDE the mask (img[outside].mean(axis=0));
    that's a fancy-index on a bool mask, which costs ~5 ms at 256×256. v1.1
    uses the global mean — for typical defect sizes (5–15% of pixels) this
    differs by <1% in colour values, and the per-call cost drops to ~50 µs.
    The downstream noise term dominates either way."""
    bg = img.mean(axis=(0, 1))
    bg = np.clip(bg + rng.normal(0, 0.10, 3), 0, 1).astype(np.float32)
    noise = rng.normal(0, 0.05, img.shape).astype(np.float32)
    m3 = mask.astype(np.float32)[..., None]
    return np.clip((1.0 - m3) * img + m3 * (bg[None, None, :] + noise), 0, 1).astype(
        np.float32
    )


def appearance_mold(
    img: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Greenish or grayish fuzzy texture."""
    if rng.random() < 0.5:
        base = np.array([0.40, 0.50, 0.35])  # greenish
    else:
        base = np.array([0.50, 0.50, 0.50])  # grayish
    H, W = mask.shape
    texture = base[None, None, :] + rng.normal(0, 0.10, (H, W, 3))
    texture = np.clip(texture, 0, 1).astype(np.float32)
    alpha = float(rng.uniform(0.50, 0.85))
    m3 = mask.astype(np.float32)[..., None]
    return ((1.0 - alpha * m3) * img + (alpha * m3) * texture).astype(np.float32)


def appearance_contamination(
    img: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Hard dark dots — pest-damage flavour."""
    color = np.clip(np.array([0.10, 0.08, 0.05]) + rng.normal(0, 0.03, 3), 0, 1).astype(
        np.float32
    )
    alpha = float(rng.uniform(0.70, 1.00))
    m3 = mask.astype(np.float32)[..., None]
    return ((1.0 - alpha * m3) * img + (alpha * m3) * color[None, None, :]).astype(
        np.float32
    )


APPEARANCE_GENERATORS: dict[str, Callable] = {
    "scratch": appearance_scratch,
    "crack": appearance_crack,
    "dent": appearance_dent,
    "bulge": appearance_bulge,
    "stain": appearance_stain,
    "fragment": appearance_fragment,
    "mold": appearance_mold,
    "contamination": appearance_contamination,
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-class defect spec + composer
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DefectSpec:
    """One defect family wired up for a class. The shape and appearance
    generators are paired here so the injector can apply them with one
    call."""

    family: str
    shape_fn: Callable
    appearance_fn: Callable


def make_class_defect_specs(
    class_name: str,
    taxonomy: dict[str, list[str]],
    fallback: tuple[str, ...] = _FALLBACK_FAMILIES,
) -> list[DefectSpec]:
    """Build the list of DefectSpec to use for one class. Pulls the
    families from the taxonomy if present, otherwise from the fallback
    mixture."""
    fams = taxonomy.get(class_name) or list(fallback)
    out: list[DefectSpec] = []
    for f in fams:
        if f in SHAPE_GENERATORS and f in APPEARANCE_GENERATORS:
            out.append(
                DefectSpec(
                    family=f,
                    shape_fn=SHAPE_GENERATORS[f],
                    appearance_fn=APPEARANCE_GENERATORS[f],
                )
            )
    if not out:
        for f in SHAPE_GENERATORS:
            out.append(
                DefectSpec(
                    family=f,
                    shape_fn=SHAPE_GENERATORS[f],
                    appearance_fn=APPEARANCE_GENERATORS[f],
                )
            )
    return out


def inject_defects(
    img01: np.ndarray,
    specs: list[DefectSpec],
    rng: np.random.Generator,
    n_defects_range: tuple[int, int] = (1, 3),
    min_mask_pixels: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply n ∈ [n_min, n_max] class-appropriate defects to a clean
    image. Image and mask are emitted as a pair.

    img01     : (H, W, 3) float32 in [0, 1]
    specs     : per-class list of DefectSpec
    rng       : numpy Generator (advancing across calls)
    Returns   : (image, mask) — image is float32 (H, W, 3) in [0, 1];
                mask is float32 (H, W) in {0, 1}.
    """
    if not specs:
        return img01.copy(), np.zeros(img01.shape[:2], dtype=np.float32)
    H, W = img01.shape[:2]
    n = int(rng.integers(n_defects_range[0], n_defects_range[1] + 1))
    accum = np.zeros((H, W), dtype=np.float32)
    out = img01.copy()
    for _ in range(n):
        spec = specs[int(rng.integers(0, len(specs)))]
        # A generator can legitimately draw nothing — a scratch sampled
        # entirely outside the frame, say. That is a skip, not an error. A
        # generator that *raises* is a bug, so it is no longer swallowed:
        # the original caught bare Exception here and silently produced
        # clean images labelled as defective.
        m = spec.shape_fn(H, W, rng)
        if m.sum() < min_mask_pixels:
            continue
        out = spec.appearance_fn(out, m, rng)
        accum = np.maximum(accum, m)
    return out.astype(np.float32), accum.astype(np.float32)


FAMILIES: tuple[str, ...] = tuple(SHAPE_GENERATORS)
