"""Metrics tests: sanity bounds and known-value checks (no LPIPS network
weights needed for these -- LPIPSMetric is tested separately, skipped if
the lpips package can't fetch its own pretrained weights offline).
"""

from __future__ import annotations

import torch

from frameforge.metrics import (
    compute_all_metrics,
    count_parameters,
    estimate_gflops,
    psnr,
    rgb_to_y,
    ssim,
    temporal_error,
)
from frameforge.models.student import FrameForgeNet


def test_psnr_identical_is_inf():
    x = torch.rand(1, 3, 16, 16)
    assert psnr(x, x).item() == float("inf")


def test_psnr_known_value():
    # MSE = 0.01 exactly -> PSNR = 10*log10(1/0.01) = 20 dB
    pred = torch.zeros(1, 3, 4, 4)
    target = torch.full((1, 3, 4, 4), 0.1)
    value = psnr(pred, target).item()
    assert abs(value - 20.0) < 1e-3


def test_ssim_identical_is_one():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 32, 32)
    value = ssim(x, x).item()
    assert abs(value - 1.0) < 1e-4


def test_ssim_bounded():
    torch.manual_seed(0)
    pred = torch.rand(1, 3, 32, 32)
    target = torch.rand(1, 3, 32, 32)
    value = ssim(pred, target).item()
    assert -1.0 <= value <= 1.0


def test_rgb_to_y_shape_and_range():
    x = torch.rand(2, 3, 8, 8)
    y = rgb_to_y(x)
    assert y.shape == (2, 1, 8, 8)
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_temporal_error_zero_for_static_sequence():
    frame = torch.rand(1, 3, 16, 16)
    seq = frame.repeat(3, 1, 1, 1)
    lr_mv = torch.zeros(3, 2, 8, 8)
    err = temporal_error(seq, lr_mv)
    assert err.item() < 1e-6


def test_count_parameters_matches_manual_sum():
    model = FrameForgeNet(variant="single_frame", channels=8, num_blocks=1)
    expected = sum(p.numel() for p in model.parameters())
    assert count_parameters(model) == expected


def test_estimate_gflops_positive_and_scales_with_resolution():
    model = FrameForgeNet(variant="single_frame", channels=8, num_blocks=1)
    small = estimate_gflops(model, lr_h=32, lr_w=32)
    large = estimate_gflops(model, lr_h=64, lr_w=64)
    assert small > 0
    assert large > small * 3  # ~4x pixels -> roughly 4x FLOPs


def test_compute_all_metrics_bundle_keys():
    t, h, w = 3, 16, 16
    pred = torch.rand(t, 3, h, w)
    target = torch.rand(t, 3, h, w)
    lr_mv = torch.zeros(t, 2, h // 2, w // 2)
    results = compute_all_metrics(pred, target, lr_mv=lr_mv)
    for key in ["psnr_rgb", "ssim_rgb", "psnr_y", "ssim_y",
                "temporal_error_pred", "temporal_error_gt"]:
        assert key in results
        assert isinstance(results[key], float)
