"""Load a full rendered sequence (LR or HR) into ready-to-use tensors.

Shared by ``dataset.py`` (on-the-fly loading) and ``cache.py`` (offline
preprocessing to .pt files) so there is exactly one place that implements
"raw EXR -> tonemapped/normalized tensor".
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch

from frameforge.colorspace import normalize_depth, tonemap_and_encode
from frameforge.data.exr_io import blender_vector_to_backward_flow, load_frame_passes

_FRAME_RE = re.compile(r"frame_(\d+)\.exr$")


def list_frame_files(seq_dir: Path) -> list[Path]:
    def _frame_index(p: Path) -> int:
        return int(_FRAME_RE.search(p.name).group(1))

    files = sorted(seq_dir.glob("frame_*.exr"), key=_frame_index)
    if not files:
        raise FileNotFoundError(f"no frame_*.exr files found in {seq_dir}")
    return files


def load_lr_sequence(
    seq_dir: Path, channel_map: dict, tonemap_cfg: dict
) -> dict[str, torch.Tensor]:
    """Load an LR sequence directory into (rgb, depth, mv) tensors.

    Returns:
        dict with:
          "rgb": (T, 3, H, W) tonemapped sRGB in [0, 1]
          "depth": (T, 1, H, W) normalized to [0, 1], shared min/max across
              the whole sequence (CLAUDE.md 3.3: per-sequence, not per-frame)
          "mv": (T, 2, H, W) backward flow, pixel units, (dx, dy). Frame 0's
              vector is forced to zero (no previous frame exists), matching
              student.py's documented "first frame = no temporal input" rule.
    """
    files = list_frame_files(seq_dir)
    combined, depth, vec_prev = [], [], []
    for f in files:
        passes = load_frame_passes(f, channel_map)
        combined.append(passes["combined"])
        depth.append(passes["depth"])
        vec_prev.append(passes["vector_to_previous"])

    combined_t = torch.from_numpy(np.stack(combined)).permute(0, 3, 1, 2).float()  # (T,3,H,W)
    depth_t = torch.from_numpy(np.stack(depth)).permute(0, 3, 1, 2).float()  # (T,1,H,W)

    rgb = tonemap_and_encode(
        combined_t, method=tonemap_cfg["method"], white_point=tonemap_cfg.get("white_point")
    )
    depth_norm = normalize_depth(depth_t)

    mv_frames = [blender_vector_to_backward_flow(v).squeeze(0) for v in vec_prev]  # each (2,H,W)
    mv = torch.stack(mv_frames, dim=0)  # (T,2,H,W)
    mv[0] = 0.0  # no previous frame for the sequence's first frame

    return {"rgb": rgb, "depth": depth_norm, "mv": mv}


def load_hr_sequence(seq_dir: Path, channel_map: dict, tonemap_cfg: dict) -> torch.Tensor:
    """Load an HR sequence directory's Combined pass as ground-truth RGB.

    Returns: (T, 3, H, W) tonemapped sRGB in [0, 1].
    """
    files = list_frame_files(seq_dir)
    combined = [load_frame_passes(f, channel_map)["combined"] for f in files]
    combined_t = torch.from_numpy(np.stack(combined)).permute(0, 3, 1, 2).float()
    return tonemap_and_encode(
        combined_t, method=tonemap_cfg["method"], white_point=tonemap_cfg.get("white_point")
    )
