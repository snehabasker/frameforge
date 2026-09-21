"""EXR I/O tests, including the spec-required warp-consistency check
(CLAUDE.md 3.2) run against committed synthetic fixture EXRs.

This is the closest we can get to the spec's literal requirement without a
real Blender render: it exercises the full path (read multilayer EXR -> pull
out the Vector pass -> convert via the (currently-identity, flagged-unverified)
Blender adapter -> warp -> compare against the next real frame) end to end,
on a fixture whose ground-truth camera pan is known exactly because we
synthesized it. It proves the pipeline plumbing and the warp math are
correct. It does NOT prove Blender's real Vector pass matches our assumed
sign/axis/unit convention -- that still needs a real render (docs/DECISIONS.md D4).
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from frameforge.data.exr_io import blender_vector_to_backward_flow, load_frame_passes
from frameforge.warp import warp

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _channel_map():
    with open(REPO_ROOT / "configs" / "data.yaml") as f:
        cfg = yaml.safe_load(f)
    return cfg["exr_channel_map"]


def test_load_frame_passes_shapes():
    cmap = _channel_map()
    frame = load_frame_passes(FIXTURES / "sequences/synthetic_pan_00/lr/frame_0001.exr", cmap)
    assert frame["combined"].shape == (36, 64, 3)
    assert frame["depth"].shape == (36, 64, 1)
    assert frame["vector_to_previous"].shape == (36, 64, 2)
    assert frame["vector_to_next"].shape == (36, 64, 2)


def test_warp_consistency_on_lr_fixture_sequence():
    """CLAUDE.md 3.2's required test, run on the synthetic LR fixture."""
    cmap = _channel_map()
    seq_dir = FIXTURES / "sequences/synthetic_pan_00/lr"

    frame0 = load_frame_passes(seq_dir / "frame_0000.exr", cmap)
    frame1 = load_frame_passes(seq_dir / "frame_0001.exr", cmap)

    frame0_rgb = torch.from_numpy(frame0["combined"]).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    frame1_rgb = torch.from_numpy(frame1["combined"]).permute(2, 0, 1).unsqueeze(0)

    flow = blender_vector_to_backward_flow(frame1["vector_to_previous"])
    warped = warp(frame0_rgb, flow)

    # Interior region only (the pan shifts content by 2px in x, so the
    # rightmost couple of columns see disocclusion/border effects).
    margin = 6
    err = (warped - frame1_rgb).abs()[:, :, margin:-margin, margin:-margin]
    mean_err = err.mean().item()
    assert mean_err < 5e-3, f"warp-consistency mean abs error too high: {mean_err}"


def test_warp_consistency_on_hr_fixture_sequence():
    cmap = _channel_map()
    seq_dir = FIXTURES / "sequences/synthetic_pan_00/hr"

    frame0 = load_frame_passes(seq_dir / "frame_0000.exr", cmap)
    frame1 = load_frame_passes(seq_dir / "frame_0001.exr", cmap)

    frame0_rgb = torch.from_numpy(frame0["combined"]).permute(2, 0, 1).unsqueeze(0)
    frame1_rgb = torch.from_numpy(frame1["combined"]).permute(2, 0, 1).unsqueeze(0)

    flow = blender_vector_to_backward_flow(frame1["vector_to_previous"])
    warped = warp(frame0_rgb, flow)

    margin = 12
    err = (warped - frame1_rgb).abs()[:, :, margin:-margin, margin:-margin]
    mean_err = err.mean().item()
    assert mean_err < 5e-3, f"warp-consistency mean abs error too high: {mean_err}"
