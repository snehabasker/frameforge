"""scripts/extract_failure_crops.py tests (find_worst_crops logic)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from extract_failure_crops import find_worst_crops  # noqa: E402


def test_finds_the_single_high_error_region():
    t, c, h, w = 1, 3, 32, 32
    pred = torch.zeros(t, c, h, w)
    target = torch.zeros(t, c, h, w)
    target[0, :, 20:28, 20:28] = 1.0  # a small high-error hotspot

    crops = find_worst_crops(pred, target, crop_size=8, top_k=1)
    assert len(crops) == 1
    frame_idx, y0, x0, err = crops[0]
    assert frame_idx == 0
    assert 16 <= y0 <= 24 and 16 <= x0 <= 24  # overlaps the hotspot
    assert err > 0.5


def test_top_k_crops_are_non_overlapping():
    t, c, h, w = 1, 3, 64, 64
    pred = torch.zeros(t, c, h, w)
    target = torch.rand(t, c, h, w)
    crops = find_worst_crops(pred, target, crop_size=8, top_k=5)
    assert len(crops) == 5
    for i, a in enumerate(crops):
        for b in crops[i + 1:]:
            assert not (a[0] == b[0] and abs(a[1] - b[1]) < 8 and abs(a[2] - b[2]) < 8)


def test_zero_error_returns_low_values():
    pred = torch.rand(1, 3, 16, 16)
    target = pred.clone()
    crops = find_worst_crops(pred, target, crop_size=8, top_k=2)
    for _, _, _, err in crops:
        assert err < 1e-6
