"""Export a fixed-seed model to ONNX and dump its Python (PyTorch)
recurrent-inference output on a real sequence, for the C++ engine to
compare itself against (CLAUDE.md 7.4: "Engine output vs Python reference
on a fixed sequence (tolerance-based)").

Writes, under --out:
  model.onnx
  reference_output.bin  (raw float32, T*3*H*W values, NCHW per frame,
                          frames concatenated in order)
  meta.txt              ("T H W" on one line)

Usage: python scripts/export_reference_sequence.py --seq-dir DIR --out DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from frameforge.data.dataset import FrameSequenceDataset
from frameforge.export_onnx import export_model
from frameforge.models.student import FrameForgeNet
from frameforge.train import run_unroll


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-dir", required=True, help="a single sequence's parent dir "
                                                           "(passed to FrameSequenceDataset)")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=2)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(args.seed)
    dataset = FrameSequenceDataset(args.seq_dir, cfg["exr_channel_map"], cfg["tonemap"])
    sample = dataset[0]

    model = FrameForgeNet(variant="temporal", channels=args.channels, num_blocks=args.num_blocks)
    model.eval()

    with torch.no_grad():
        reference_out = run_unroll(model, sample)  # (T, 3, H, W)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    export_model(
        model, out_dir / "model.onnx",
        h=sample.lr_rgb.shape[-2], w=sample.lr_rgb.shape[-1],
    )

    reference_out.numpy().astype("float32").tofile(out_dir / "reference_output.bin")
    t, _, h, w = reference_out.shape
    (out_dir / "meta.txt").write_text(f"{t} {h} {w}\n")
    print(f"wrote model.onnx + reference_output.bin ({t}x3x{h}x{w}) to {out_dir}")


if __name__ == "__main__":
    main()
