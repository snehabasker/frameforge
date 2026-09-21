"""EXR -> .pt cache round-trip test."""

from __future__ import annotations

from pathlib import Path

import yaml

from frameforge.data.cache import CachedSequenceDataset, build_cache

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _cfg():
    with open(REPO_ROOT / "configs" / "data.yaml") as f:
        return yaml.safe_load(f)


def test_build_cache_and_load_round_trip(tmp_path):
    cfg = _cfg()
    written = build_cache(
        FIXTURES / "sequences", tmp_path, cfg["exr_channel_map"], cfg["tonemap"]
    )
    assert len(written) == 1
    assert written[0].exists()

    ds = CachedSequenceDataset(tmp_path)
    assert len(ds) == 1
    lr_rgb, lr_depth, lr_mv, hr_rgb = ds[0]
    assert lr_rgb.shape == (3, 3, 36, 64)
    assert hr_rgb.shape == (3, 3, 72, 128)
    assert lr_rgb.dtype.is_floating_point
