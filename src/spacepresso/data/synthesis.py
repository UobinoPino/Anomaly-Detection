"""Synthetic anomaly generation.

Three detectors train on artificial defects painted onto normal images, and
each shipped its own copy of the machinery: DRAEM (Perlin-masked texture
blending), CutPaste (rectangular patch transplants), and GLASS (Perlin masks
again, on the GPU). The Perlin implementation in particular existed twice, in
NumPy and in torch, with different octave conventions.

One NumPy implementation lives here, plus the torch one GLASS needs for
on-device batch synthesis.

Determinism: every function takes an explicit ``rng``. The originals reached
for a module-level generator that ``worker_init_fn`` reseeded, which meant
augmentation diversity depended on a global whose lifetime nobody owned.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch

__all__ = [
    "cutpaste",
    "cutpaste_scar",
    "draem_corruption",
    "fractal_noise",
    "perlin_2d",
    "perlin_batch_torch",
    "random_texture",
    "sample_perlin_mask",
]


def _smoothstep(t: npt.NDArray) -> npt.NDArray:
    """Perlin's quintic fade: zero first *and* second derivative at 0 and 1."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin_2d(
    shape: tuple[int, int], res: tuple[int, int], rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    """Classic 2-D Perlin noise in roughly [-1, 1].

    ``shape`` must be divisible by ``res``; :func:`sample_perlin_mask` pads to
    a multiple and crops, so callers rarely need to care.
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    step = (shape[0] // res[0], shape[1] // res[1])
    grid = (
        np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1.0
    )

    angles = 2 * np.pi * rng.random((res[0] + 1, res[1] + 1))
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    gradients = gradients.repeat(step[0], 0).repeat(step[1], 1)

    g00 = gradients[: -step[0], : -step[1]]
    g10 = gradients[step[0] :, : -step[1]]
    g01 = gradients[: -step[0], step[1] :]
    g11 = gradients[step[0] :, step[1] :]

    n00 = np.sum(grid * g00, 2)
    n10 = np.sum((grid - np.array([1, 0])) * g10, 2)
    n01 = np.sum((grid - np.array([0, 1])) * g01, 2)
    n11 = np.sum((grid - np.array([1, 1])) * g11, 2)

    fade = _smoothstep(grid)
    lower = n00 * (1 - fade[..., 0]) + fade[..., 0] * n10
    upper = n01 * (1 - fade[..., 0]) + fade[..., 0] * n11
    return np.sqrt(2) * ((1 - fade[..., 1]) * lower + fade[..., 1] * upper)


def fractal_noise(
    shape: tuple[int, int],
    res: tuple[int, int],
    rng: np.random.Generator,
    octaves: int = 4,
    persistence: float = 0.5,
) -> npt.NDArray[np.float64]:
    """Sum of Perlin octaves at doubling frequency and decaying amplitude."""
    noise = np.zeros(shape)
    frequency, amplitude = 1, 1.0
    for _ in range(octaves):
        noise += amplitude * perlin_2d(
            shape, (frequency * res[0], frequency * res[1]), rng
        )
        frequency *= 2
        amplitude *= persistence
    return noise


def sample_perlin_mask(
    height: int, width: int, rng: np.random.Generator
) -> npt.NDArray[np.float32]:
    """A binary blob mask covering roughly 15–40% of the image.

    Two frequency scales are mixed so the mask has both large regions and
    fine structure; the threshold percentile is sampled per call so coverage
    varies across the training set rather than being fixed.
    """

    def octave(scale: int) -> npt.NDArray:
        padded = ((height + scale * 8 - 1) // (scale * 8)) * (scale * 8)
        return fractal_noise(
            (padded, padded), (scale, scale), rng, octaves=3, persistence=0.5
        )[:height, :width]

    noise = 0.5 * octave(int(rng.choice([2, 4, 8]))) + 0.5 * octave(
        int(rng.choice([8, 16]))
    )
    mask = (noise > np.percentile(noise, rng.uniform(60, 85))).astype(np.float32)
    if rng.random() < 0.5:
        mask = np.rot90(mask, k=int(rng.integers(1, 4))).copy()
    return mask


def random_texture(
    height: int, width: int, rng: np.random.Generator
) -> npt.NDArray[np.float32]:
    """A flat or two-colour-gradient patch with light pixel noise, in [0, 1]."""
    if rng.random() < 0.5:
        start = rng.random(3).astype(np.float32)
        end = rng.random(3).astype(np.float32)
        ramp = (
            np.linspace(0, 1, width, dtype=np.float32)[None, :, None]
            if rng.random() < 0.5
            else np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
        )
        texture = start[None, None, :] * (1 - ramp) + end[None, None, :] * ramp
        texture = np.broadcast_to(texture, (height, width, 3)).copy()
    else:
        colour = rng.random(3).astype(np.float32)
        texture = np.broadcast_to(colour[None, None, :], (height, width, 3)).copy()

    noise = rng.normal(0.0, 0.03, size=(height, width, 3)).astype(np.float32)
    return np.clip(texture + noise, 0.0, 1.0)


def draem_corruption(
    image: npt.NDArray[np.float32], rng: np.random.Generator
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Blend a random texture into a Perlin-masked region.

    Returns ``(corrupted, mask)``. The blend strength ``beta`` is sampled in
    [0.15, 1.0] so the training set spans barely-visible to fully-replaced —
    a detector trained only on opaque defects never learns the subtle ones.
    """
    height, width = image.shape[:2]
    mask = sample_perlin_mask(height, width, rng)
    texture = random_texture(height, width, rng)
    beta = float(rng.uniform(0.15, 1.0))

    mask3 = mask[..., None]
    corrupted = (1.0 - beta * mask3) * image + (beta * mask3) * texture
    return np.clip(corrupted, 0.0, 1.0).astype(np.float32), mask.astype(np.float32)


def _jitter(patch: npt.NDArray, rng: np.random.Generator, strength: float):
    factor = 1.0 + rng.uniform(-strength, strength, size=3).astype(np.float32)
    return np.clip(patch.astype(np.float32) * factor, 0, 255).astype(patch.dtype)


def cutpaste(
    image: npt.NDArray,
    rng: np.random.Generator,
    *,
    area_ratio: tuple[float, float] = (0.02, 0.15),
    aspect_ratio: tuple[float, float] = (0.3, 3.3),
    jitter: float = 0.1,
) -> npt.NDArray:
    """Copy a random rectangular region and paste it elsewhere.

    The transplanted patch is real texture from the same image, so the
    classifier cannot win by detecting "this doesn't look like the material" —
    it has to notice the discontinuity at the seam, which is what a real
    defect looks like.
    """
    height, width = image.shape[:2]
    area = height * width * rng.uniform(*area_ratio)
    ratio = rng.uniform(*aspect_ratio)

    patch_h = int(round(np.sqrt(area / ratio)))
    patch_w = int(round(np.sqrt(area * ratio)))
    patch_h = int(np.clip(patch_h, 1, height - 1))
    patch_w = int(np.clip(patch_w, 1, width - 1))

    src_y = int(rng.integers(0, height - patch_h))
    src_x = int(rng.integers(0, width - patch_w))
    patch = image[src_y : src_y + patch_h, src_x : src_x + patch_w].copy()
    if jitter > 0:
        patch = _jitter(patch, rng, jitter)

    dst_y = int(rng.integers(0, height - patch_h))
    dst_x = int(rng.integers(0, width - patch_w))
    out = image.copy()
    out[dst_y : dst_y + patch_h, dst_x : dst_x + patch_w] = patch
    return out


def cutpaste_scar(
    image: npt.NDArray,
    rng: np.random.Generator,
    *,
    width_range: tuple[int, int] = (10, 25),
    height_range: tuple[int, int] = (2, 16),
    rotation: tuple[float, float] = (-45.0, 45.0),
    jitter: float = 0.1,
) -> npt.NDArray:
    """The thin, rotated variant — long scratches rather than blobs.

    Scars matter disproportionately here: the Spacepresso classes include
    scratched metal, and a model trained only on square CutPaste patches
    transfers poorly to elongated defects.
    """
    from PIL import Image

    img_h, img_w = image.shape[:2]
    scar_w = int(rng.integers(*width_range))
    scar_h = int(rng.integers(*height_range))
    scar_w = min(scar_w, img_w - 1)
    scar_h = min(scar_h, img_h - 1)

    src_y = int(rng.integers(0, img_h - scar_h))
    src_x = int(rng.integers(0, img_w - scar_w))
    patch = image[src_y : src_y + scar_h, src_x : src_x + scar_w].copy()
    if jitter > 0:
        patch = _jitter(patch, rng, jitter)

    rotated = Image.fromarray(patch.astype(np.uint8)).rotate(
        float(rng.uniform(*rotation)), expand=True
    )
    rotated_array = np.asarray(rotated)
    rot_h, rot_w = rotated_array.shape[:2]
    if rot_h >= img_h or rot_w >= img_w:
        return image.copy()

    dst_y = int(rng.integers(0, img_h - rot_h))
    dst_x = int(rng.integers(0, img_w - rot_w))
    out = image.copy()
    region = out[dst_y : dst_y + rot_h, dst_x : dst_x + rot_w]
    # The rotation pads with black; only paste where the patch is real.
    occupied = rotated_array.sum(axis=-1) > 0
    region[occupied] = rotated_array[occupied]
    return out


def perlin_batch_torch(
    batch: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    octaves: int = 4,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Batched Perlin-like noise on the GPU, ``(B, H, W)`` in roughly [-1, 1].

    GLASS synthesises anomalies inside the training loop, so pulling each mask
    back from a NumPy generator would stall the device every iteration. This
    sums upsampled random lattices — not identical to the gradient-based
    construction above, but the same spectral character, and it never leaves
    the GPU.
    """
    total = torch.zeros(batch, 1, height, width, device=device)
    amplitude = 1.0
    for octave in range(octaves):
        cells = 2 ** (octave + 1)
        lattice = torch.rand(
            batch, 1, cells + 1, cells + 1, device=device, generator=generator
        )
        upsampled = torch.nn.functional.interpolate(
            lattice, size=(height, width), mode="bicubic", align_corners=True
        )
        total = total + amplitude * (upsampled * 2 - 1)
        amplitude *= 0.5
    return total.squeeze(1)
