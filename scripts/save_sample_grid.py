"""Save a sample grid image of the dataset pipeline's output (CLAUDE.md M1
acceptance: "a sample grid image is saved to docs/").

Rows = frames of a sequence. Columns = LR (bicubic-upsampled for display),
HR ground truth, and a depth visualization -- i.e. exactly what
`FrameSequenceDataset` hands the training loop, laid out for a human to
sanity-check by eye.

Usage: python scripts/save_sample_grid.py [--root DIR] [--out PATH]
Defaults to the committed synthetic fixture, since no real rendered dataset
exists in this environment yet (see docs/DECISIONS.md D0). Re-run with
--root pointing at a real rendered dataset once one exists.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from frameforge.data.dataset import FrameSequenceDataset

REPO_ROOT = Path(__file__).resolve().parents[1]


def _to_uint8(img_chw: torch.Tensor) -> Image.Image:
    arr = (img_chw.clamp(0, 1) * 255.0).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def build_grid(root: Path, out_path: Path, config_path: Path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ds = FrameSequenceDataset(root, cfg["exr_channel_map"], cfg["tonemap"])
    sample = ds[0]
    lr_rgb, lr_depth, lr_mv, hr_rgb = sample
    t, _, hr_h, hr_w = hr_rgb.shape

    lr_upsampled = F.interpolate(lr_rgb, size=(hr_h, hr_w), mode="bicubic", align_corners=False)
    depth_vis = lr_depth.repeat(1, 3, 1, 1)
    depth_vis_up = F.interpolate(depth_vis, size=(hr_h, hr_w), mode="nearest")

    pad = 4
    tile_w, tile_h = hr_w, hr_h
    cols = 3  # LR bicubic | HR ground truth | depth
    grid_w = cols * tile_w + (cols + 1) * pad
    grid_h = t * tile_h + (t + 1) * pad
    grid = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))

    for frame_idx in range(t):
        y = pad + frame_idx * (tile_h + pad)
        tiles = [lr_upsampled[frame_idx], hr_rgb[frame_idx], depth_vis_up[frame_idx]]
        for col_idx, tile in enumerate(tiles):
            x = pad + col_idx * (tile_w + pad)
            grid.paste(_to_uint8(tile), (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    print(f"wrote {out_path} ({grid_w}x{grid_h}, {t} frames x {cols} cols: "
          f"LR-bicubic | HR-ground-truth | depth)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPO_ROOT / "tests/fixtures/sequences"))
    parser.add_argument("--out", default=str(REPO_ROOT / "docs/sample_grid.png"))
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/data.yaml"))
    args = parser.parse_args()
    build_grid(Path(args.root), Path(args.out), Path(args.config))


if __name__ == "__main__":
    main()
