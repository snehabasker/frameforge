"""FrameSequenceDataset tests against the synthetic fixture."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from frameforge.data.dataset import FrameSequenceDataset, discover_sequences

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _cfg():
    with open(REPO_ROOT / "configs" / "data.yaml") as f:
        return yaml.safe_load(f)


def test_discover_sequences_finds_fixture():
    found = discover_sequences(FIXTURES / "sequences")
    names = [(scene, seq) for scene, seq, _ in found]
    assert ("synthetic_pan_00", "synthetic_pan_00") in names


def test_dataset_returns_expected_shapes_and_unpacks_as_tuple():
    cfg = _cfg()
    ds = FrameSequenceDataset(
        FIXTURES / "sequences", cfg["exr_channel_map"], cfg["tonemap"],
    )
    assert len(ds) == 1
    sample = ds[0]
    lr_rgb, lr_depth, lr_mv, hr_rgb = sample  # spec's exact tuple form

    assert lr_rgb.shape == (3, 3, 36, 64)
    assert lr_depth.shape == (3, 1, 36, 64)
    assert lr_mv.shape == (3, 2, 36, 64)
    assert hr_rgb.shape == (3, 3, 72, 128)

    assert lr_rgb.min() >= 0.0 and lr_rgb.max() <= 1.0
    assert lr_depth.min() >= 0.0 and lr_depth.max() <= 1.0
    # first frame has no previous frame -> mv forced to zero
    assert (lr_mv[0] == 0.0).all()


def test_dataset_unroll_window_shorter_than_sequence():
    cfg = _cfg()
    ds = FrameSequenceDataset(
        FIXTURES / "sequences", cfg["exr_channel_map"], cfg["tonemap"],
        unroll_length=2, seed=0,
    )
    sample = ds[0]
    assert sample.lr_rgb.shape[0] == 2
    assert sample.hr_rgb.shape[0] == 2


def test_dataset_crop():
    cfg = _cfg()
    ds = FrameSequenceDataset(
        FIXTURES / "sequences", cfg["exr_channel_map"], cfg["tonemap"],
        crop_size=16, seed=0,
    )
    sample = ds[0]
    assert sample.lr_rgb.shape[-2:] == (16, 16)
    assert sample.hr_rgb.shape[-2:] == (32, 32)  # x2 scale


def test_dataset_missing_scene_raises():
    with pytest.raises(FileNotFoundError):
        FrameSequenceDataset(
            FIXTURES / "sequences", _cfg()["exr_channel_map"], _cfg()["tonemap"],
            scenes=["nonexistent_scene"],
        )
