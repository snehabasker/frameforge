"""Motion-compensated warping via ``grid_sample``.

This module owns exactly one convention, used everywhere else in the
codebase (model forward pass, temporal loss, evaluation, the C++ engine's
expectations): a **backward flow**, ``target -> source``, in **pixel units**
at the resolution of the tensor being warped, channel order **(dx, dy)**
i.e. (x-displacement, y-displacement).

Concretely: to reconstruct frame ``t`` from frame ``t-1``, ``flow[..., y, x]``
gives the pixel offset such that::

    frame_t[y, x] ~= frame_tm1[y + flow_y[y, x], x + flow_x[y, x]]

This is the convention PyTorch's ``grid_sample`` wants once converted to
normalized ``[-1, 1]`` coordinates, which :func:`warp` does internally.

Blender's raw Vector render pass is a *different*, EXR-specific encoding
(screen-space motion to the previous/next frame, in an as-yet-empirically-
unconfirmed sign/axis/unit convention -- see module ``frameforge.data.exr_io``
and ``docs/DECISIONS.md`` D-series entries for the M1 data milestone). Never
feed a raw Vector pass into :func:`warp` directly; convert it first with
``frameforge.data.exr_io.blender_vector_to_backward_flow``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def make_base_grid(height: int, width: int, device=None, dtype=None) -> torch.Tensor:
    """Pixel-center coordinate grid, shape (H, W, 2), channel order (x, y)."""
    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1)  # (H, W, 2)


def flow_to_normalized_grid(flow: torch.Tensor) -> torch.Tensor:
    """Convert a pixel-unit backward flow to a ``grid_sample`` sampling grid.

    Args:
        flow: (B, 2, H, W) tensor, channel order (dx, dy), pixel units, in the
            target frame's coordinate system.

    Returns:
        (B, H, W, 2) tensor of sampling coordinates in [-1, 1], as expected by
        ``torch.nn.functional.grid_sample`` with ``align_corners=False``.
    """
    if flow.dim() != 4 or flow.shape[1] != 2:
        raise ValueError(f"flow must be (B, 2, H, W), got {tuple(flow.shape)}")
    b, _, h, w = flow.shape
    base = make_base_grid(h, w, device=flow.device, dtype=flow.dtype)  # (H, W, 2)
    base = base.unsqueeze(0).expand(b, -1, -1, -1)  # (B, H, W, 2)
    flow_hw2 = flow.permute(0, 2, 3, 1)  # (B, H, W, 2), (dx, dy)
    sample_px = base + flow_hw2  # pixel coords to sample from, (x, y)

    # align_corners=False: pixel center i maps to normalized ((i + 0.5) / N) * 2 - 1
    sample_x = (sample_px[..., 0] + 0.5) / w * 2.0 - 1.0
    sample_y = (sample_px[..., 1] + 0.5) / h * 2.0 - 1.0
    return torch.stack((sample_x, sample_y), dim=-1)


def warp(
    source: torch.Tensor,
    flow: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "border",
) -> torch.Tensor:
    """Warp ``source`` into the target frame using a backward pixel flow.

    Args:
        source: (B, C, H, W) tensor, e.g. frame t-1.
        flow: (B, 2, H, W) tensor, backward flow target->source, pixel units,
            channel order (dx, dy). Must be at the same H, W as ``source``
            (upsample the flow yourself first if it was computed at a
            different resolution -- see ``student.py``'s use for the
            previous-output warp, which upsamples LR motion vectors to HR
            before warping the HR previous output).
        mode: passed to ``grid_sample`` ("bilinear" or "nearest").
        padding_mode: passed to ``grid_sample``. "border" clamps
            out-of-frame samples to the edge, which is the right default for
            disocclusion regions at frame borders (interior disocclusions
            are handled separately via a validity mask, not padding).

    Returns:
        (B, C, H, W) tensor: source warped to align with the target frame.
    """
    if source.dim() != 4:
        raise ValueError(f"source must be (B, C, H, W), got {tuple(source.shape)}")
    if source.shape[-2:] != flow.shape[-2:]:
        raise ValueError(
            f"source spatial size {tuple(source.shape[-2:])} != "
            f"flow spatial size {tuple(flow.shape[-2:])}"
        )
    grid = flow_to_normalized_grid(flow)
    return F.grid_sample(
        source, grid, mode=mode, padding_mode=padding_mode, align_corners=False
    )


def validity_mask(
    flow: torch.Tensor,
    forward_flow: torch.Tensor | None = None,
    consistency_threshold: float = 1.0,
) -> torch.Tensor:
    """Disocclusion/validity mask from forward-backward flow consistency.

    A pixel is valid if warping the backward flow by the forward flow and
    back roughly cancels out, i.e. the two flows agree on where matter moved.
    Without a forward flow (e.g. only Blender's "to previous frame" vector is
    available for a given pair), falls back to an in-bounds-only mask: 1.0
    where the backward-flow sample lands inside the source frame, 0.0 at the
    border (a conservative proxy -- true disocclusion detection additionally
    needs the depth pass, wired up in ``frameforge.data.dataset``).

    Args:
        flow: (B, 2, H, W) backward flow, pixel units, (dx, dy).
        forward_flow: optional (B, 2, H, W) forward flow at the source frame.
        consistency_threshold: max allowed round-trip pixel error to call a
            pixel valid, only used when ``forward_flow`` is given.

    Returns:
        (B, 1, H, W) tensor in [0, 1] (soft in-bounds case) or {0, 1}
        (consistency case).
    """
    b, _, h, w = flow.shape
    grid = flow_to_normalized_grid(flow)  # (B, H, W, 2) in [-1, 1]
    in_bounds = (
        (grid[..., 0] >= -1.0)
        & (grid[..., 0] <= 1.0)
        & (grid[..., 1] >= -1.0)
        & (grid[..., 1] <= 1.0)
    ).float().unsqueeze(1)

    if forward_flow is None:
        return in_bounds

    warped_forward = warp(forward_flow, flow, mode="bilinear", padding_mode="border")
    round_trip = flow + warped_forward  # should be ~0 if consistent
    error = torch.linalg.vector_norm(round_trip, dim=1, keepdim=True)
    consistent = (error <= consistency_threshold).float()
    return consistent * in_bounds
