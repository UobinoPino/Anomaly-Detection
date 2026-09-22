"""Building blocks shared by the segmentation-style detectors."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["UNet", "dice_loss", "focal_loss", "ssim_loss"]


def _conv_block(c_in: int, c_out: int, max_groups: int = 8) -> nn.Sequential:
    # GroupNorm rather than BatchNorm: batches are small (8) and per-class, so
    # batch statistics are noisy and shift between classes.
    #
    # The group count is the largest divisor of c_out up to max_groups. Fixing
    # it at 8 made the whole network fail to construct for any base width not
    # a multiple of 8, which is a needless constraint on a tunable.
    groups = math.gcd(c_out, max_groups) or 1
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.GroupNorm(num_groups=groups, num_channels=c_out),
        nn.ReLU(inplace=True),
    )


class _DoubleConv(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(_conv_block(c_in, c_out), _conv_block(c_out, c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Down(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(c_in, c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Up(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = _DoubleConv(c_in + c_out, c_out)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.conv(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    """Six-level U-Net, used for both roles: 3→3 and 6→2 channels."""

    def __init__(self, in_channels: int, out_channels: int, base: int = 32) -> None:
        super().__init__()
        widths = [base * 2**i for i in range(6)]
        self.inc = _DoubleConv(in_channels, widths[0])
        self.downs = nn.ModuleList(
            _Down(widths[i], widths[i + 1]) for i in range(len(widths) - 1)
        )
        self.ups = nn.ModuleList(
            _Up(widths[i + 1], widths[i]) for i in reversed(range(len(widths) - 1))
        )
        self.outc = nn.Conv2d(widths[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        out = skips[-1]
        for up, skip in zip(self.ups, reversed(skips[:-1]), strict=True):
            out = up(out, skip)
        return self.outc(out)


# ─────────────────────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────────────────────
def ssim_loss(x: torch.Tensor, y: torch.Tensor, window: int = 11, sigma: float = 1.5):
    """1 − SSIM, computed in fp32.

    The stability constants are 1e-4 and 9e-4; in fp16 those are close to the
    subnormal range and the loss goes to NaN within a few hundred steps. The
    caller keeps this out of autocast.
    """
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=1.0, neginf=0.0)
    y = torch.nan_to_num(y.float(), nan=0.0, posinf=1.0, neginf=0.0)

    channels = x.shape[1]
    half = window // 2
    coords = torch.arange(window, device=x.device, dtype=x.dtype) - half
    gauss = torch.exp(-(coords**2) / (2 * sigma * sigma))
    gauss = gauss / (gauss.sum() + 1e-8)
    kernel = (
        (gauss[:, None] * gauss[None, :])
        .clamp_min(1e-8)
        .expand(channels, 1, window, window)
        .contiguous()
    )

    def blur(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t, kernel, padding=half, groups=channels)

    mu_x, mu_y = blur(x), blur(y)
    mu_x2, mu_y2, mu_xy = mu_x.pow(2), mu_y.pow(2), mu_x * mu_y
    var_x = (blur(x * x) - mu_x2).clamp(min=1e-8)
    var_y = (blur(y * y) - mu_y2).clamp(min=1e-8)
    cov = (blur(x * y) - mu_xy).clamp(-1e6, 1e6)

    c1, c2 = 1e-4, 9e-4
    numerator = (2 * mu_xy + c1) * (2 * cov + c2)
    denominator = ((mu_x2 + mu_y2 + c1) * (var_x + var_y + c2)).clamp_min(1e-8)
    return 1.0 - torch.nan_to_num((numerator / denominator).mean(), nan=1.0)


def focal_loss(
    logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float = 0.5
) -> torch.Tensor:
    """Focal loss over the two-class segmentation head.

    Defects cover a small fraction of the image, so plain cross-entropy is
    dominated by easy background pixels. The ``(1 - p_t)^gamma`` factor
    down-weights those; ``alpha`` additionally reweights the positive class.
    """
    log_probs = F.log_softmax(logits, dim=1)
    target = target.long()
    log_p_t = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    p_t = log_p_t.exp()
    weight = torch.where(
        target == 1, torch.full_like(p_t, alpha), torch.full_like(p_t, 1.0 - alpha)
    )
    return (-weight * (1.0 - p_t) ** gamma * log_p_t).mean()


def dice_loss(
    logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Soft Dice on the positive class.

    Focal loss optimises per-pixel accuracy and tolerates a mask that is
    slightly the wrong *shape*; Dice penalises overlap directly, which keeps
    the predicted region tight around the defect.
    """
    probability = F.softmax(logits, dim=1)[:, 1]
    positive = (target > 0).float()
    intersection = (probability * positive).sum(dim=(1, 2))
    union = probability.sum(dim=(1, 2)) + positive.sum(dim=(1, 2))
    return (1.0 - (2.0 * intersection + eps) / (union + eps)).mean()
