"""Fixed tonemap + sRGB encoding, and depth normalization (CLAUDE.md 3.3).

The SAME transform is used everywhere -- feeding models, computing metrics,
and saving preview images -- so nothing here may depend on model outputs or
per-call state beyond what's passed in.
"""

from __future__ import annotations

import torch

_SRGB_A = 0.055


def reinhard_tonemap(linear_hdr: torch.Tensor, white_point: float | None = None) -> torch.Tensor:
    """Reinhard tonemap, linear HDR -> linear [0, 1].

    Simple Reinhard: ``x / (1 + x)``, applied per-channel.
    Extended Reinhard (if ``white_point`` given): ``x * (1 + x / Lwhite^2) / (1 + x)``,
    which allows values above 1 to still map to 1 rather than asymptoting.
    """
    if white_point is None:
        return linear_hdr / (1.0 + linear_hdr)
    lwhite2 = white_point * white_point
    return (linear_hdr * (1.0 + linear_hdr / lwhite2)) / (1.0 + linear_hdr)


def linear_to_srgb(linear: torch.Tensor) -> torch.Tensor:
    """IEC 61966-2-1 sRGB OETF. Input assumed in [0, 1] linear light."""
    linear = linear.clamp(min=0.0)
    low = linear * 12.92
    high = (1.0 + _SRGB_A) * torch.pow(linear.clamp(min=1e-8), 1.0 / 2.4) - _SRGB_A
    return torch.where(linear <= 0.0031308, low, high).clamp(0.0, 1.0)


def srgb_to_linear(srgb: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`linear_to_srgb`, for completeness/round-trip tests."""
    srgb = srgb.clamp(0.0, 1.0)
    low = srgb / 12.92
    high = torch.pow((srgb + _SRGB_A) / (1.0 + _SRGB_A), 2.4)
    return torch.where(srgb <= 0.04045, low, high)


def tonemap_and_encode(
    linear_hdr: torch.Tensor, method: str = "reinhard", white_point: float | None = None
) -> torch.Tensor:
    """The one fixed transform: linear HDR -> tonemapped sRGB in [0, 1].

    Used identically for model inputs/targets and for every metric.
    """
    if method != "reinhard":
        raise ValueError(f"Unknown tonemap method {method!r}; only 'reinhard' is implemented")
    tonemapped_linear = reinhard_tonemap(linear_hdr, white_point=white_point)
    return linear_to_srgb(tonemapped_linear)


def normalize_depth(depth: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-sequence min-max depth normalization to [0, 1].

    Args:
        depth: (T, 1, H, W) or (1, H, W) or (H, W) tensor of raw Z-depth for
            ONE sequence (all frames normalized together with shared min/max,
            so depth stays temporally consistent within a sequence -- a
            per-frame normalization would make static geometry's depth value
            drift frame to frame, which would corrupt the temporal-consistency
            signal the model is supposed to learn from).
        eps: numerical floor for the min-max range.

    Returns:
        Same shape, values in [0, 1]. Infinite depth (background/sky, which
        Cycles' Z pass reports as a very large or inf value) is clamped to
        the finite max before normalizing, so the sky doesn't dominate the
        range.
    """
    finite = torch.isfinite(depth)
    if not torch.any(finite):
        return torch.zeros_like(depth)
    finite_vals = depth[finite]
    d_min = finite_vals.min()
    d_max = finite_vals.max()
    clamped = torch.where(finite, depth, d_max)
    return (clamped - d_min) / (d_max - d_min + eps)
