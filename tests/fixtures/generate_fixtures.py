"""Generate tiny synthetic multilayer-EXR fixtures for tests and CI.

No Blender required: builds a small analytic "scene" (a smooth pattern plus
a depth gradient) undergoing a known, constant camera pan, and writes it out
in the same multilayer-EXR channel layout ``frameforge.data.exr_io`` expects
(see configs/data.yaml's exr_channel_map). Native LR and HR versions are each
evaluated directly from the analytic function at their own resolution
(mirroring the real pipeline's "render natively, don't downsample" rule --
CLAUDE.md 3.1) rather than one being a resize of the other.

Run: python tests/fixtures/generate_fixtures.py
Output is committed to the repo (tiny: a few small frames), so CI and local
tests never need to regenerate it -- but re-running is safe and deterministic
(fixed seed, no randomness at all actually: purely analytic).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from frameforge.data.exr_io import write_exr_channels  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent
VIEW_LAYER = "ViewLayer"

# (width, height) -- intentionally tiny, and NOT a clean multiple relationship
# beyond the required x2, to catch accidental hardcoded-size bugs.
LR_SIZE = (64, 36)
HR_SIZE = (128, 72)
NUM_FRAMES = 3
PAN_PER_FRAME = (2.0, -1.0)  # (dx, dy) pixels/frame, AT LR RESOLUTION


def _field(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A smooth analytic "scene": three offset sinusoids, one per RGB channel."""
    r = 0.5 + 0.5 * np.sin(x / 9.0) * np.cos(y / 7.0)
    g = 0.5 + 0.5 * np.sin(x / 13.0 + 1.0) * np.cos(y / 11.0 + 0.5)
    b = 0.5 + 0.5 * np.sin((x + y) / 15.0 + 2.0)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _depth_field(x: np.ndarray, y: np.ndarray, w: float) -> np.ndarray:
    """A synthetic depth gradient (near at center, far at edges), panning
    with the same camera motion as the color field."""
    cx, cy = w / 2.0, w / 2.0 * 9 / 16
    d = 1.0 + np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / 20.0
    return d.astype(np.float32)


def _render_frame(size: tuple[int, int], frame_idx: int, pan: tuple[float, float]):
    w, h = size
    ys, xs = np.meshgrid(
        np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij"
    )
    # Camera pans by `pan` pixels/frame: scene content at time t sits at the
    # position it would occupy at t=0 shifted by frame_idx * pan.
    shift_x = frame_idx * pan[0]
    shift_y = frame_idx * pan[1]
    combined = _field(xs - shift_x, ys - shift_y)
    depth = _depth_field(xs - shift_x, ys - shift_y, w)
    return combined, depth


def _write_sequence(size: tuple[int, int], scale_from_lr: float, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = size
    pan = (PAN_PER_FRAME[0] * scale_from_lr, PAN_PER_FRAME[1] * scale_from_lr)

    for t in range(NUM_FRAMES):
        combined, depth = _render_frame(size, t, pan)

        # Vector pass: exact (backward-flow-shaped) ground truth, since we
        # constructed the pan ourselves. "to previous" = -pan (the offset
        # from this frame's pixel back to where that content was at t-1).
        vec_to_prev = np.full((h, w, 2), 0.0, dtype=np.float32)
        vec_to_next = np.full((h, w, 2), 0.0, dtype=np.float32)
        if t > 0:
            vec_to_prev[..., 0] = -pan[0]
            vec_to_prev[..., 1] = -pan[1]
        if t < NUM_FRAMES - 1:
            vec_to_next[..., 0] = pan[0]
            vec_to_next[..., 1] = pan[1]

        channels = {
            f"{VIEW_LAYER}.Combined.R": combined[..., 0],
            f"{VIEW_LAYER}.Combined.G": combined[..., 1],
            f"{VIEW_LAYER}.Combined.B": combined[..., 2],
            f"{VIEW_LAYER}.Depth.Z": depth,
            f"{VIEW_LAYER}.Vector.X": vec_to_prev[..., 0],
            f"{VIEW_LAYER}.Vector.Y": vec_to_prev[..., 1],
            f"{VIEW_LAYER}.Vector.Z": vec_to_next[..., 0],
            f"{VIEW_LAYER}.Vector.W": vec_to_next[..., 1],
        }
        write_exr_channels(out_dir / f"frame_{t:04d}.exr", channels)


def main():
    seq_dir = FIXTURES_DIR / "sequences" / "synthetic_pan_00"
    _write_sequence(LR_SIZE, scale_from_lr=1.0, out_dir=seq_dir / "lr")
    _write_sequence(HR_SIZE, scale_from_lr=2.0, out_dir=seq_dir / "hr")
    print(f"wrote {NUM_FRAMES} LR + {NUM_FRAMES} HR frames to {seq_dir}")


if __name__ == "__main__":
    main()
