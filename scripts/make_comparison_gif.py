"""Build the side-by-side comparison GIF (CLAUDE.md M6/section 10: "bicubic
| FSR1 | FrameForge | GT, via ffmpeg script").

For each frame, renders a labeled [bicubic | FSR1 | FrameForge | ground
truth] strip as a PNG, then calls ffmpeg (must be on PATH) to assemble the
strips into an animated GIF.

Usage:
  python scripts/make_comparison_gif.py --data-root DIR --checkpoint C.pt \
      --fsr1-cli path/to/fsr1_cli[.exe] --out docs/comparison.gif
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from frameforge.data.dataset import FrameSequenceDataset  # noqa: E402
from frameforge.evaluate import fsr1_baseline  # noqa: E402
from frameforge.models.student import FrameForgeNet  # noqa: E402
from frameforge.train import run_unroll  # noqa: E402

LABEL_HEIGHT = 16


def _to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def _labeled_strip(panels: list[tuple[str, torch.Tensor]]) -> Image.Image:
    tiles = [_to_pil(t) for _, t in panels]
    w, h = tiles[0].size
    pad = 2
    strip = Image.new(
        "RGB", (len(tiles) * w + (len(tiles) + 1) * pad, h + LABEL_HEIGHT + 2 * pad),
        (15, 15, 15),
    )
    draw = ImageDraw.Draw(strip)
    for i, (label, _) in enumerate(panels):
        x = pad + i * (w + pad)
        strip.paste(tiles[i], (x, LABEL_HEIGHT + pad))
        draw.text((x + 2, 2), label, fill=(255, 255, 255))
    return strip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(REPO_ROOT / "tests/fixtures/sequences"))
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/data.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fsr1-cli", required=True)
    parser.add_argument("--out", default=str(REPO_ROOT / "docs/comparison.gif"))
    parser.add_argument("--fps", type=int, default=2)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset = FrameSequenceDataset(args.data_root, cfg["exr_channel_map"], cfg["tonemap"])
    sample = dataset[0]

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = FrameForgeNet(
        variant=ckpt["variant"], channels=ckpt["channels"], num_blocks=ckpt["num_blocks"]
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        frameforge_out = run_unroll(model, sample)
    bicubic_out = F.interpolate(
        sample.lr_rgb, scale_factor=2, mode="bicubic", align_corners=False
    ).clamp(0, 1)
    fsr1_out = fsr1_baseline(sample, args.fsr1_cli)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for t in range(sample.lr_rgb.shape[0]):
            strip = _labeled_strip([
                ("bicubic", bicubic_out[t]),
                ("FSR1", fsr1_out[t]),
                ("FrameForge", frameforge_out[t]),
                ("ground truth", sample.hr_rgb[t]),
            ])
            strip.save(tmp_path / f"frame_{t:04d}.png")

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-framerate", str(args.fps),
            "-i", str(tmp_path / "frame_%04d.png"),
            "-vf", "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
