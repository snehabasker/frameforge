"""Evaluation metrics (CLAUDE.md section 6).

All metrics operate on tonemapped sRGB tensors in [0, 1] (the same
transform used everywhere, frameforge.colorspace) -- "computed after the
fixed tonemap" per spec, not on linear HDR.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from frameforge.losses import ground_truth_temporal_error

# ITU-R BT.601 luma weights (matches the "Y channel" convention most SR
# papers report; documented explicitly since CLAUDE.md doesn't pin one).
_LUMA_WEIGHTS = torch.tensor([0.299, 0.587, 0.114])


def rgb_to_y(rgb: torch.Tensor) -> torch.Tensor:
    """(..., 3, H, W) sRGB in [0, 1] -> (..., 1, H, W) luma."""
    weights = _LUMA_WEIGHTS.to(rgb.device, rgb.dtype).view(1, 3, 1, 1)
    if rgb.dim() == 3:
        weights = weights.squeeze(0)
    return (rgb * weights).sum(dim=-3, keepdim=True)


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    if mse.item() == 0.0:
        return torch.tensor(float("inf"))
    return 10.0 * torch.log10(max_val ** 2 / mse)


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g.outer(g)
    return kernel_2d


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Standard single-scale SSIM, computed per-channel then averaged.

    Args:
        pred, target: (B, C, H, W), values in [0, 1].
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    c = pred.shape[1]
    kernel = _gaussian_kernel(window_size).to(pred.device, pred.dtype)
    kernel = kernel.expand(c, 1, window_size, window_size)
    pad = window_size // 2

    mu_p = F.conv2d(pred, kernel, padding=pad, groups=c)
    mu_t = F.conv2d(target, kernel, padding=pad, groups=c)
    mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t

    var_p = F.conv2d(pred * pred, kernel, padding=pad, groups=c) - mu_p2
    var_t = F.conv2d(target * target, kernel, padding=pad, groups=c) - mu_t2
    covar = F.conv2d(pred * target, kernel, padding=pad, groups=c) - mu_pt

    c1, c2 = (0.01 ** 2), (0.03 ** 2)
    ssim_map = ((2 * mu_pt + c1) * (2 * covar + c2)) / (
        (mu_p2 + mu_t2 + c1) * (var_p + var_t + c2)
    )
    return ssim_map.mean()


def temporal_error(sequence_rgb: torch.Tensor, lr_mv: torch.Tensor) -> torch.Tensor:
    """Mean |O_t - warp(O_{t-1})| over valid pixels (CLAUDE.md 6).

    Report this ALONGSIDE `frameforge.losses.ground_truth_temporal_error`
    on the same sequence's HR ground truth -- the gap between the two is
    the actual signal (a model that's simply blurry can look
    "temporally stable" by this metric alone).
    """
    return ground_truth_temporal_error(sequence_rgb, lr_mv)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def estimate_gflops(model: torch.nn.Module, lr_h: int = 540, lr_w: int = 960) -> float:
    """Rough GFLOPs estimate at a given LR input size via a forward hook
    that sums 2x MACs for every Conv2d encountered (dominant cost for this
    architecture; PixelShuffle/warp/elementwise ops are comparatively
    negligible and omitted for simplicity -- documented, not hidden).
    """
    total_flops = [0]
    hooks = []

    def hook(module, inputs, output):
        out_h, out_w = output.shape[-2:]
        k_h, k_w = module.kernel_size
        in_c = module.in_channels // module.groups
        out_c = module.out_channels
        macs = out_h * out_w * out_c * in_c * k_h * k_w
        total_flops[0] += 2 * macs  # 1 MAC = 2 FLOPs

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(hook))

    try:
        model.eval()
        lr_rgb = torch.zeros(1, 3, lr_h, lr_w)
        kwargs = {}
        if getattr(model, "use_depth", False):
            kwargs["lr_depth"] = torch.zeros(1, 1, lr_h, lr_w)
        if getattr(model, "use_temporal", False):
            kwargs["lr_mv"] = torch.zeros(1, 2, lr_h, lr_w)
            kwargs["prev_hr_output"] = torch.zeros(1, 3, lr_h * model.scale, lr_w * model.scale)
        model(lr_rgb, **kwargs)
    finally:
        for h in hooks:
            h.remove()

    return total_flops[0] / 1e9


class LPIPSMetric:
    """Thin lazy wrapper around the `lpips` package (AlexNet backbone, per
    CLAUDE.md 6). Lazy-imported/instantiated so importing this module never
    requires downloading LPIPS' own pretrained weights unless actually used.
    """

    def __init__(self):
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            import lpips as lpips_pkg

            self._model = lpips_pkg.LPIPS(net="alex")
            self._model.eval()

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target: (B, 3, H, W) in [0, 1]. lpips expects [-1, 1]."""
        self._ensure_loaded()
        pred_scaled = pred * 2.0 - 1.0
        target_scaled = target * 2.0 - 1.0
        return self._model(pred_scaled, target_scaled).mean()


def compute_all_metrics(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    lr_mv: torch.Tensor | None = None,
    lpips_metric: LPIPSMetric | None = None,
) -> dict[str, float]:
    """One-shot metric bundle for a (T, 3, H, W) prediction vs. target pair."""
    results = {
        "psnr_rgb": psnr(pred_rgb, target_rgb).item(),
        "ssim_rgb": ssim(pred_rgb, target_rgb).item(),
        "psnr_y": psnr(rgb_to_y(pred_rgb), rgb_to_y(target_rgb)).item(),
        "ssim_y": ssim(rgb_to_y(pred_rgb), rgb_to_y(target_rgb)).item(),
    }
    if lpips_metric is not None:
        results["lpips"] = lpips_metric(pred_rgb, target_rgb).item()
    if lr_mv is not None and pred_rgb.shape[0] > 1:
        results["temporal_error_pred"] = temporal_error(pred_rgb, lr_mv).item()
        results["temporal_error_gt"] = temporal_error(target_rgb, lr_mv).item()
    return results
