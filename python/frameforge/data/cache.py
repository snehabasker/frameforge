"""Offline EXR -> cached tensor preprocessing (CLAUDE.md 3.3).

Converts each discovered sequence to a single .pt file holding the same
tensors `dataset.py` would otherwise recompute from EXR on every access.
Meant to be run once per rendered dataset; `FrameSequenceDataset` stays the
on-the-fly / no-cache-needed path used directly against small data (e.g. the
test fixtures), while `CachedSequenceDataset` here reads the precomputed
.pt files for fast repeated access during actual training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset

from frameforge.data.dataset import discover_sequences
from frameforge.data.preprocess import load_hr_sequence, load_lr_sequence


def build_cache(
    root: str | Path,
    out_dir: str | Path,
    channel_map: dict,
    tonemap_cfg: dict,
    scenes: list[str] | None = None,
) -> list[Path]:
    """Preprocess every sequence under `root` and write one .pt file each.

    Returns the list of written cache file paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for scene, sequence, seq_dir in discover_sequences(Path(root), scenes=scenes):
        lr = load_lr_sequence(seq_dir / "lr", channel_map, tonemap_cfg)
        hr_rgb = load_hr_sequence(seq_dir / "hr", channel_map, tonemap_cfg)
        payload = {
            "lr_rgb": lr["rgb"].half(),
            "lr_depth": lr["depth"].half(),
            "lr_mv": lr["mv"].half(),
            "hr_rgb": hr_rgb.half(),
            "scene": scene,
            "sequence": sequence,
        }
        out_path = out_dir / f"{scene}__{sequence}.pt"
        torch.save(payload, out_path)
        written.append(out_path)
    return written


class CachedSequenceDataset(Dataset):
    """Reads .pt files written by :func:`build_cache`."""

    def __init__(self, cache_dir: str | Path):
        self.files = sorted(Path(cache_dir).glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"no cached sequences (*.pt) found in {cache_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        payload = torch.load(self.files[idx], weights_only=True)
        return (
            payload["lr_rgb"].float(),
            payload["lr_depth"].float(),
            payload["lr_mv"].float(),
            payload["hr_rgb"].float(),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="dir of scene/sequence/{lr,hr} EXRs")
    parser.add_argument("--out", required=True, help="output dir for .pt cache files")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    written = build_cache(args.root, args.out, cfg["exr_channel_map"], cfg["tonemap"])
    print(f"wrote {len(written)} cache file(s) to {args.out}")


if __name__ == "__main__":
    main()
