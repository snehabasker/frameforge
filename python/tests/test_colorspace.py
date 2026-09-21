"""Tonemap/colorspace tests: round-trip correctness and depth normalization."""

from __future__ import annotations

import torch

from frameforge.colorspace import (
    linear_to_srgb,
    normalize_depth,
    reinhard_tonemap,
    srgb_to_linear,
    tonemap_and_encode,
)


def test_srgb_round_trip():
    torch.manual_seed(0)
    x = torch.rand(1000)
    recovered = srgb_to_linear(linear_to_srgb(x))
    assert torch.allclose(recovered, x, atol=1e-4)


def test_srgb_known_points():
    # sRGB(0) = 0, sRGB(1) = 1, exactly.
    x = torch.tensor([0.0, 1.0])
    y = linear_to_srgb(x)
    assert torch.allclose(y, torch.tensor([0.0, 1.0]), atol=1e-6)


def test_reinhard_maps_to_unit_interval_and_is_monotonic():
    x = torch.linspace(0.0, 1000.0, steps=50)
    y = reinhard_tonemap(x)
    assert torch.all(y >= 0.0) and torch.all(y < 1.0)
    assert torch.all(torch.diff(y) > 0), "Reinhard tonemap must be monotonic"


def test_tonemap_and_encode_is_bounded():
    torch.manual_seed(0)
    hdr = torch.rand(2, 3, 8, 8) * 50.0  # simulate bright HDR values
    out = tonemap_and_encode(hdr)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_normalize_depth_range_and_finite_clamp():
    depth = torch.tensor([1.0, 2.0, 3.0, float("inf")]).reshape(1, 1, 2, 2)
    normed = normalize_depth(depth)
    finite_mask = torch.isfinite(depth)
    assert normed[finite_mask].min().item() == 0.0
    assert abs(normed[finite_mask].max().item() - 1.0) < 1e-4
    # inf pixel clamped to the finite max, i.e. normalized value 1.0
    assert torch.isclose(normed[~finite_mask], torch.tensor(1.0), atol=1e-5)


def test_normalize_depth_all_inf_returns_zeros():
    depth = torch.full((1, 1, 4, 4), float("inf"))
    normed = normalize_depth(depth)
    assert torch.all(normed == 0.0)
