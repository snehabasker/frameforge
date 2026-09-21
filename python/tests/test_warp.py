"""Warp-consistency tests (CLAUDE.md section 3.2).

These establish that ``frameforge.warp.warp`` correctly implements its
documented backward-flow convention (target -> source, pixel units, (dx, dy)
channel order) using synthetic camera pans with known ground truth -- no
Blender required. Confirming that *Blender's raw Vector pass* maps onto this
convention with the right sign/axis/scale is a separate, still-pending step
that needs a real render; see docs/DECISIONS.md.
"""

from __future__ import annotations

import torch

from frameforge.warp import validity_mask, warp


def _integer_shift_ground_truth(frame_tm1: torch.Tensor, dx: int, dy: int):
    """Shift frame_tm1 by (dx, dy) pixels via exact slicing (no interpolation).

    Returns (frame_t, valid_mask) where valid_mask marks pixels that came
    from inside frame_tm1 (i.e. non-disoccluded, non-border pixels).
    """
    _, _, h, w = frame_tm1.shape
    frame_t = torch.zeros_like(frame_tm1)
    valid = torch.zeros(1, 1, h, w)

    dst_y0, dst_y1 = max(0, dy), h + min(0, dy)
    dst_x0, dst_x1 = max(0, dx), w + min(0, dx)
    src_y0, src_y1 = max(0, -dy), h + min(0, -dy)
    src_x0, src_x1 = max(0, -dx), w + min(0, -dx)

    frame_t[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = frame_tm1[:, :, src_y0:src_y1, src_x0:src_x1]
    valid[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = 1.0
    return frame_t, valid


def test_warp_recovers_integer_pixel_pan():
    """A camera pan shifting content right by dx, up by dy pixels."""
    torch.manual_seed(0)
    h, w = 64, 64
    frame_tm1 = torch.rand(1, 3, h, w)
    dx, dy = 5, -3  # content moves +5 in x, -3 in y (i.e. up) from t-1 to t

    frame_t, valid = _integer_shift_ground_truth(frame_tm1, dx, dy)

    # Backward flow: target(t) -> source(t-1) is the *inverse* of the pan.
    flow = torch.zeros(1, 2, h, w)
    flow[:, 0] = -dx
    flow[:, 1] = -dy

    warped = warp(frame_tm1, flow)
    err = (warped - frame_t).abs() * valid
    mean_err = err.sum() / (valid.sum() * frame_tm1.shape[1])
    assert mean_err.item() < 1e-4, f"warp mean abs error too high: {mean_err.item()}"


def test_warp_wrong_sign_fails_the_same_check():
    """Sanity: flipping the flow sign should NOT pass, proving the test is
    actually discriminating and not vacuously true."""
    torch.manual_seed(0)
    h, w = 64, 64
    frame_tm1 = torch.rand(1, 3, h, w)
    dx, dy = 5, -3
    frame_t, valid = _integer_shift_ground_truth(frame_tm1, dx, dy)

    wrong_flow = torch.zeros(1, 2, h, w)
    wrong_flow[:, 0] = dx  # wrong sign
    wrong_flow[:, 1] = dy

    warped = warp(frame_tm1, wrong_flow)
    err = (warped - frame_t).abs() * valid
    mean_err = err.sum() / (valid.sum() * frame_tm1.shape[1])
    assert mean_err.item() > 1e-2, "expected the wrong-sign flow to visibly disagree"


def test_warp_subpixel_pan_against_analytic_ground_truth():
    """Sub-pixel pan on a smooth analytic field, ground truth evaluated
    directly (not via grid_sample), so this doesn't just test itself."""
    h, w = 96, 96
    ys = torch.arange(h, dtype=torch.float32)
    xs = torch.arange(w, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    def field(x, y):
        return torch.sin(x / 11.3) * torch.cos(y / 9.7) * 0.5 + 0.5

    frame_tm1 = field(grid_x, grid_y).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    dx, dy = 2.4, -1.7  # sub-pixel pan
    frame_t = field(grid_x - dx, grid_y - dy).unsqueeze(0).unsqueeze(0)

    flow = torch.zeros(1, 2, h, w)
    flow[:, 0] = -dx
    flow[:, 1] = -dy

    warped = warp(frame_tm1, flow)
    # Interior only: analytic field is smooth everywhere but grid_sample
    # border padding still differs slightly from the analytic extrapolation
    # right at the edges.
    margin = 8
    interior = warped[:, :, margin:-margin, margin:-margin]
    interior_gt = frame_t[:, :, margin:-margin, margin:-margin]
    mean_err = (interior - interior_gt).abs().mean()
    assert mean_err.item() < 5e-3, f"sub-pixel warp mean abs error too high: {mean_err.item()}"


def test_validity_mask_in_bounds_fallback():
    h, w = 32, 32
    flow = torch.zeros(1, 2, h, w)
    flow[:, 0] = 100.0  # every pixel samples far out of bounds
    mask = validity_mask(flow)
    assert mask.shape == (1, 1, h, w)
    assert torch.all(mask == 0.0)

    flow_zero = torch.zeros(1, 2, h, w)
    mask_zero = validity_mask(flow_zero)
    assert torch.all(mask_zero == 1.0)
