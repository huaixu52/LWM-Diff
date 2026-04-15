"""
Out-of-Distribution (OOD) Corruption Benchmark for Ultrasound Images.

Provides corruption functions that are **independent** of the training
augmentation pipeline (``apply_ultrasound_augmentation``).  The goal is to
evaluate model robustness to noise types / severity levels *never seen*
during training.

Design follows the ImageNet-C paradigm:
  - Multiple corruption types
  - 5 severity levels per type (1 = mild … 5 = extreme)
  - Deterministic per-sample when seeded, stochastic otherwise

Usage
-----
>>> from datasets.ood_corruptions import apply_ood_corruption, CORRUPTION_TYPES
>>> corrupted_img = apply_ood_corruption(img_tensor, "gaussian_noise", severity=3)
>>> # Or apply a random corruption:
>>> corrupted_img = apply_ood_corruption(img_tensor, "random", severity=3)
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CORRUPTION_TYPES: List[str] = [
    "gaussian_noise",
    "salt_pepper",
    "speckle_extreme",
    "motion_blur",
    "defocus_blur",
    "signal_dropout",
    "patch_occlusion",
    "resolution_degradation",
    "brightness_shift",
    "contrast_shift",
]

_SEVERITY_RANGE = range(1, 6)  # 1..5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_ood_corruption(
    img: torch.Tensor,
    corruption_type: str = "gaussian_noise",
    severity: int = 3,
) -> torch.Tensor:
    """Apply a single OOD corruption to an image tensor ``(C, H, W)`` in [0,1].

    Parameters
    ----------
    img : torch.Tensor
        Image tensor of shape ``(C, H, W)`` with values in ``[0, 1]``.
    corruption_type : str
        One of :data:`CORRUPTION_TYPES`, or ``"random"`` to pick one at random.
    severity : int
        Severity level in ``{1, 2, 3, 4, 5}``.

    Returns
    -------
    torch.Tensor
        Corrupted image, clamped to ``[0, 1]``.
    """
    if severity not in _SEVERITY_RANGE:
        raise ValueError(f"severity must be in 1..5, got {severity}")

    if corruption_type == "random":
        corruption_type = random.choice(CORRUPTION_TYPES)

    fn = _DISPATCH.get(corruption_type)
    if fn is None:
        raise ValueError(
            f"Unknown corruption_type={corruption_type!r}. "
            f"Choose from {CORRUPTION_TYPES} or 'random'."
        )

    img = img.float().clone()
    if img.max() > 1.0:
        img = img / 255.0
    img = img.clamp(0.0, 1.0)

    return fn(img, severity).clamp(0.0, 1.0)


def get_all_corruption_configs() -> List[Dict[str, Union[str, int]]]:
    """Return a list of ``{type, severity}`` dicts for the full benchmark."""
    configs = []
    for ctype in CORRUPTION_TYPES:
        for sev in _SEVERITY_RANGE:
            configs.append({"type": ctype, "severity": sev})
    return configs


# ---------------------------------------------------------------------------
# Individual corruption implementations
# ---------------------------------------------------------------------------

def _gaussian_noise(img: torch.Tensor, severity: int) -> torch.Tensor:
    """Additive Gaussian noise.  σ ∈ {0.02, 0.05, 0.10, 0.15, 0.22}."""
    sigma = [0.02, 0.05, 0.10, 0.15, 0.22][severity - 1]
    return img + torch.randn_like(img) * sigma


def _salt_pepper(img: torch.Tensor, severity: int) -> torch.Tensor:
    """Salt-and-pepper (impulse) noise.  Fraction ∈ {0.01, 0.03, 0.06, 0.10, 0.18}."""
    frac = [0.01, 0.03, 0.06, 0.10, 0.18][severity - 1]
    mask = torch.rand_like(img)
    out = img.clone()
    out[mask < frac / 2] = 0.0
    out[mask > 1.0 - frac / 2] = 1.0
    return out


def _speckle_extreme(img: torch.Tensor, severity: int) -> torch.Tensor:
    """Multiplicative speckle noise at extreme levels (beyond training range).
    σ ∈ {0.20, 0.35, 0.50, 0.70, 1.00}."""
    sigma = [0.20, 0.35, 0.50, 0.70, 1.00][severity - 1]
    return img * (1.0 + torch.randn_like(img) * sigma)


def _motion_blur(img: torch.Tensor, severity: int) -> torch.Tensor:
    """Directional motion blur with random angle.
    Kernel size ∈ {5, 9, 15, 21, 29}."""
    ksize = [5, 9, 15, 21, 29][severity - 1]
    angle = random.uniform(0, 360)

    # Build 1D motion kernel then rotate
    kernel = torch.zeros(ksize, ksize, dtype=img.dtype, device=img.device)
    kernel[ksize // 2, :] = 1.0 / ksize

    # Rotate kernel
    theta = math.radians(angle)
    cos_a, sin_a = math.cos(theta), math.sin(theta)
    # Use grid_sample to rotate the kernel
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, ksize, device=img.device),
        torch.linspace(-1, 1, ksize, device=img.device),
        indexing="ij",
    )
    rot_x = cos_a * grid_x - sin_a * grid_y
    rot_y = sin_a * grid_x + cos_a * grid_y
    grid = torch.stack([rot_x, rot_y], dim=-1).unsqueeze(0)
    kernel_rot = F.grid_sample(
        kernel.unsqueeze(0).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze()
    kernel_rot = kernel_rot / (kernel_rot.sum() + 1e-8)

    # Apply via conv2d
    C = img.shape[0]
    weight = kernel_rot.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)
    pad = ksize // 2
    return F.conv2d(
        img.unsqueeze(0), weight, padding=pad, groups=C
    ).squeeze(0)


def _defocus_blur(img: torch.Tensor, severity: int) -> torch.Tensor:
    """Disk (defocus) blur with radius ∈ {2, 3, 5, 7, 10}."""
    radius = [2, 3, 5, 7, 10][severity - 1]
    ksize = 2 * radius + 1
    # Build disk kernel
    y, x = torch.meshgrid(
        torch.arange(ksize, dtype=img.dtype, device=img.device) - radius,
        torch.arange(ksize, dtype=img.dtype, device=img.device) - radius,
        indexing="ij",
    )
    disk = ((x ** 2 + y ** 2) <= radius ** 2).float()
    disk = disk / (disk.sum() + 1e-8)

    C = img.shape[0]
    weight = disk.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)
    pad = radius
    return F.conv2d(
        img.unsqueeze(0), weight, padding=pad, groups=C
    ).squeeze(0)


def _signal_dropout(img: torch.Tensor, severity: int) -> torch.Tensor:
    """Simulate ultrasound signal dropout: random horizontal/vertical lines set to zero.
    Number of lines ∈ {3, 8, 15, 25, 40}."""
    n_lines = [3, 8, 15, 25, 40][severity - 1]
    out = img.clone()
    _, H, W = img.shape
    line_width = max(1, severity)

    for _ in range(n_lines):
        if random.random() < 0.5:
            # Horizontal line dropout
            row = random.randint(0, H - line_width)
            out[:, row : row + line_width, :] = 0.0
        else:
            # Vertical line dropout
            col = random.randint(0, W - line_width)
            out[:, :, col : col + line_width] = 0.0
    return out


def _patch_occlusion(img: torch.Tensor, severity: int) -> torch.Tensor:
    """Random rectangular patch occlusion (set to black).
    Patch fraction of image area ∈ {0.03, 0.06, 0.12, 0.20, 0.30}."""
    area_frac = [0.03, 0.06, 0.12, 0.20, 0.30][severity - 1]
    _, H, W = img.shape
    out = img.clone()

    # Possibly multiple smaller patches
    n_patches = random.randint(1, max(1, severity))
    per_patch_frac = area_frac / n_patches

    for _ in range(n_patches):
        ph = max(4, int(H * math.sqrt(per_patch_frac)))
        pw = max(4, int(W * math.sqrt(per_patch_frac)))
        top = random.randint(0, max(0, H - ph))
        left = random.randint(0, max(0, W - pw))
        out[:, top : top + ph, left : left + pw] = 0.0
    return out


def _resolution_degradation(img: torch.Tensor, severity: int) -> torch.Tensor:
    """Downsample then upsample back to original resolution.
    Scale factor ∈ {0.75, 0.50, 0.33, 0.25, 0.15}."""
    scale = [0.75, 0.50, 0.33, 0.25, 0.15][severity - 1]
    _, H, W = img.shape
    small_h, small_w = max(4, int(H * scale)), max(4, int(W * scale))
    down = F.interpolate(
        img.unsqueeze(0), size=(small_h, small_w), mode="bilinear", align_corners=False
    )
    up = F.interpolate(
        down, size=(H, W), mode="bilinear", align_corners=False
    ).squeeze(0)
    return up


def _brightness_shift(img: torch.Tensor, severity: int) -> torch.Tensor:
    """Global brightness shift (additive).
    Δ ∈ {±0.05, ±0.10, ±0.18, ±0.28, ±0.40}."""
    delta = [0.05, 0.10, 0.18, 0.28, 0.40][severity - 1]
    shift = random.uniform(-delta, delta)
    return img + shift


def _contrast_shift(img: torch.Tensor, severity: int) -> torch.Tensor:
    """Global contrast scaling around mean.
    Factor ∈ {0.85, 0.65, 0.45, 0.30, 0.15} or {1.15, 1.35, 1.55, 1.70, 1.85}."""
    low_factors = [0.85, 0.65, 0.45, 0.30, 0.15]
    high_factors = [1.15, 1.35, 1.55, 1.70, 1.85]
    if random.random() < 0.5:
        factor = low_factors[severity - 1]
    else:
        factor = high_factors[severity - 1]
    mean = img.mean()
    return mean + factor * (img - mean)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_DISPATCH = {
    "gaussian_noise": _gaussian_noise,
    "salt_pepper": _salt_pepper,
    "speckle_extreme": _speckle_extreme,
    "motion_blur": _motion_blur,
    "defocus_blur": _defocus_blur,
    "signal_dropout": _signal_dropout,
    "patch_occlusion": _patch_occlusion,
    "resolution_degradation": _resolution_degradation,
    "brightness_shift": _brightness_shift,
    "contrast_shift": _contrast_shift,
}
