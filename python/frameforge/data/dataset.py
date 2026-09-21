"""Sequence dataset: returns (lr_rgb, lr_depth, lr_mv, hr_rgb) tuples.

Directory layout expected under `root` (matches tests/fixtures/generate_fixtures.py
and the layout blender/render_sequences.py writes):

    root/
      <scene_name>/
        <sequence_name>/
          lr/frame_0000.exr, frame_0001.exr, ...
          hr/frame_0000.exr, frame_0001.exr, ...

Splitting is by scene (and within a scene, by sequence), never by frame,
per CLAUDE.md 3.1: "Split train/val/test by sequence and scene, never by
frame. Hold out one entire scene for testing."
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from frameforge.data.preprocess import load_hr_sequence, load_lr_sequence


@dataclass
class SequenceBatch:
    lr_rgb: torch.Tensor    # (T, 3, H_lr, W_lr)
    lr_depth: torch.Tensor  # (T, 1, H_lr, W_lr)
    lr_mv: torch.Tensor     # (T, 2, H_lr, W_lr)
    hr_rgb: torch.Tensor    # (T, 3, H_hr, W_hr)
    scene: str
    sequence: str

    def __iter__(self):
        # Lets callers do `lr_rgb, lr_depth, lr_mv, hr_rgb = dataset[i]`
        # per the spec's exact wording, while keeping scene/sequence around
        # for anyone who wants them via attribute access.
        return iter((self.lr_rgb, self.lr_depth, self.lr_mv, self.hr_rgb))


def discover_sequences(root: Path, scenes: list[str] | None = None) -> list[tuple[str, str, Path]]:
    """Find (scene_name, sequence_name, sequence_dir) triples under root.

    A directory counts as a sequence if it directly contains `lr/` and `hr/`
    subdirectories.
    """
    root = Path(root)
    found = []
    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if scenes is not None and scene_dir.name not in scenes:
            continue
        # Two supported layouts: root/scene/sequence/{lr,hr} or, for a
        # single-sequence scene (as in the tiny test fixtures),
        # root/scene/{lr,hr} directly.
        if (scene_dir / "lr").is_dir() and (scene_dir / "hr").is_dir():
            found.append((scene_dir.name, scene_dir.name, scene_dir))
            continue
        for seq_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
            if (seq_dir / "lr").is_dir() and (seq_dir / "hr").is_dir():
                found.append((scene_dir.name, seq_dir.name, seq_dir))
    return found


class FrameSequenceDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        channel_map: dict,
        tonemap_cfg: dict,
        scenes: list[str] | None = None,
        unroll_length: int | None = None,
        crop_size: int | None = None,
        seed: int | None = None,
    ):
        self.sequences = discover_sequences(Path(root), scenes=scenes)
        if not self.sequences:
            raise FileNotFoundError(f"no sequences found under {root} (scenes={scenes})")
        self.channel_map = channel_map
        self.tonemap_cfg = tonemap_cfg
        self.unroll_length = unroll_length
        self.crop_size = crop_size
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> SequenceBatch:
        scene, sequence, seq_dir = self.sequences[idx]
        lr = load_lr_sequence(seq_dir / "lr", self.channel_map, self.tonemap_cfg)
        hr_rgb = load_hr_sequence(seq_dir / "hr", self.channel_map, self.tonemap_cfg)

        lr_rgb, lr_depth, lr_mv = lr["rgb"], lr["depth"], lr["mv"]
        t_total = lr_rgb.shape[0]
        if hr_rgb.shape[0] != t_total:
            raise ValueError(
                f"{seq_dir}: LR has {t_total} frames but HR has {hr_rgb.shape[0]}"
            )

        if self.unroll_length is not None and t_total > self.unroll_length:
            start = self._rng.randint(0, t_total - self.unroll_length)
            end = start + self.unroll_length
            lr_rgb, lr_depth, lr_mv = lr_rgb[start:end], lr_depth[start:end], lr_mv[start:end]
            hr_rgb = hr_rgb[start:end]
            if start > 0:
                # The window's first frame DOES have a real previous frame
                # (start-1) in the full sequence; only the true sequence
                # start has no previous. load_lr_sequence already zeroed
                # index 0 of the *full* sequence's mv before we sliced, so
                # nothing further to do here -- this branch exists just to
                # document that this case is handled, not accidentally
                # re-zeroed.
                pass

        if self.crop_size is not None:
            lr_rgb, lr_depth, lr_mv, hr_rgb = _random_crop(
                lr_rgb, lr_depth, lr_mv, hr_rgb, self.crop_size, self._rng
            )

        return SequenceBatch(
            lr_rgb=lr_rgb, lr_depth=lr_depth, lr_mv=lr_mv, hr_rgb=hr_rgb,
            scene=scene, sequence=sequence,
        )


def _random_crop(lr_rgb, lr_depth, lr_mv, hr_rgb, crop_size: int, rng: random.Random):
    _, _, h, w = lr_rgb.shape
    if crop_size > h or crop_size > w:
        raise ValueError(f"crop_size {crop_size} larger than LR frame {(h, w)}")
    scale = hr_rgb.shape[-1] // w
    y0 = rng.randint(0, h - crop_size)
    x0 = rng.randint(0, w - crop_size)
    lr_rgb = lr_rgb[:, :, y0:y0 + crop_size, x0:x0 + crop_size]
    lr_depth = lr_depth[:, :, y0:y0 + crop_size, x0:x0 + crop_size]
    lr_mv = lr_mv[:, :, y0:y0 + crop_size, x0:x0 + crop_size]
    hy0, hx0, hc = y0 * scale, x0 * scale, crop_size * scale
    hr_rgb = hr_rgb[:, :, hy0:hy0 + hc, hx0:hx0 + hc]
    return lr_rgb, lr_depth, lr_mv, hr_rgb
