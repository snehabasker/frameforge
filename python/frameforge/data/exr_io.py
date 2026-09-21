"""Multilayer OpenEXR read/write for rendered LR/HR sequences.

Reading is generic (works on any multilayer EXR with dotted channel names).
The Blender-specific piece -- turning Blender's raw ``Vector`` pass into this
codebase's canonical backward-flow convention (see ``frameforge.warp``) --
is isolated in :func:`blender_vector_to_backward_flow` and is the one
function in this module whose correctness is NOT yet empirically confirmed
(see docs/DECISIONS.md, "D4. Blender EXR channel naming").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import OpenEXR
import torch


def read_exr_channels(path: str | Path) -> dict[str, np.ndarray]:
    """Read every channel of an EXR file into a flat {name: (H, W) float32} dict.

    Works for both single-layer and Blender-style multilayer EXR (channel
    names like "ViewLayer.Combined.R" are kept as-is, un-grouped).
    """
    f = OpenEXR.File(str(path), separate_channels=True)
    part = f.parts[0]
    channels: dict[str, np.ndarray] = {}
    for name, chan in part.channels.items():
        channels[name] = np.asarray(chan.pixels, dtype=np.float32)
    return channels


def group_channels_by_layer(channels: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Split {"ViewLayer.Combined.R": arr, ...} into {"ViewLayer.Combined": {"R": arr}, ...}."""
    grouped: dict[str, dict[str, np.ndarray]] = {}
    for full_name, arr in channels.items():
        if "." in full_name:
            layer, chan = full_name.rsplit(".", 1)
        else:
            layer, chan = "", full_name
        grouped.setdefault(layer, {})[chan] = arr
    return grouped


def stack_channels(layer: dict[str, np.ndarray], order: list[str]) -> np.ndarray:
    """Stack named 2D channels into a (H, W, len(order)) array, in `order`."""
    missing = [c for c in order if c not in layer]
    if missing:
        raise KeyError(f"missing channels {missing}; available: {sorted(layer)}")
    return np.stack([layer[c] for c in order], axis=-1)


def load_frame_passes(path: str | Path, channel_map: dict) -> dict[str, np.ndarray]:
    """Load one rendered frame's Combined/Depth/Vector passes as numpy arrays.

    Args:
        path: path to the multilayer EXR.
        channel_map: the ``exr_channel_map`` block from ``configs/data.yaml``.

    Returns:
        dict with keys:
          "combined": (H, W, 3) float32, linear HDR RGB
          "depth": (H, W, 1) float32, raw Z depth
          "vector_to_previous": (H, W, 2) float32, raw Blender units
          "vector_to_next": (H, W, 2) float32, raw Blender units
    """
    channels = read_exr_channels(path)
    grouped = group_channels_by_layer(channels)
    vl = channel_map["view_layer"]

    combined_layer = grouped[f"{vl}.{channel_map['combined_pass']}"]
    combined = stack_channels(combined_layer, ["R", "G", "B"])

    depth_layer = grouped[f"{vl}.{channel_map['depth_pass']}"]
    depth = stack_channels(depth_layer, ["Z"])

    vector_layer = grouped[f"{vl}.{channel_map['vector_pass']}"]
    vec_prev = stack_channels(vector_layer, channel_map["vector_to_previous_channels"])
    vec_next = stack_channels(vector_layer, channel_map["vector_to_next_channels"])

    return {
        "combined": combined,
        "depth": depth,
        "vector_to_previous": vec_prev,
        "vector_to_next": vec_next,
    }


def blender_vector_to_backward_flow(vector_to_previous: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert Blender's raw "vector to previous frame" pass to a canonical
    backward flow usable by ``frameforge.warp.warp``.

    ASSUMED (unverified) convention, per ``configs/data.yaml``: the raw pass
    already stores (dx, dy) in pixel units pointing from a pixel in frame t
    to where that surface point was in frame t-1 -- i.e. exactly this
    codebase's backward-flow convention -- so this is currently an identity
    reshape, not a real transform. This function exists so that if the real
    convention turns out to need a sign flip, an axis swap, or a unit
    rescale (e.g. Blender may store the vector in a resolution-normalized
    space rather than raw pixels), there is exactly one place to fix it.
    Flip/scale corrections belong here once confirmed against a real render
    -- see the (currently synthetic-only) test in
    ``python/tests/test_exr_io.py::test_blender_vector_convention_roundtrip_on_fixture``.

    Args:
        vector_to_previous: (H, W, 2) array/tensor, channels (X, Y) as named
            in the EXR's Vector pass.

    Returns:
        (1, 2, H, W) tensor, channel order (dx, dy), ready for
        ``frameforge.warp.warp``.
    """
    if isinstance(vector_to_previous, np.ndarray):
        vector_to_previous = torch.from_numpy(vector_to_previous.copy())
    if vector_to_previous.dim() != 3 or vector_to_previous.shape[-1] != 2:
        raise ValueError(
            f"expected (H, W, 2), got {tuple(vector_to_previous.shape)}"
        )
    flow = vector_to_previous.permute(2, 0, 1).unsqueeze(0).float()  # (1, 2, H, W)
    return flow


def write_exr_channels(
    path: str | Path, channels: dict[str, np.ndarray], compression=None
) -> None:
    """Write a flat {name: (H, W) float32} dict as a multilayer EXR.

    Used by ``tests/fixtures/generate_fixtures.py`` to build tiny synthetic
    fixtures that mimic Blender's multilayer output structure, without
    needing Blender itself.
    """
    header: dict = {"type": OpenEXR.scanlineimage}
    if compression is not None:
        header["compression"] = compression
    exr_channels = {name: np.ascontiguousarray(arr, dtype=np.float32) for name, arr in channels.items()}
    f = OpenEXR.File(header, exr_channels)
    f.write(str(path))
