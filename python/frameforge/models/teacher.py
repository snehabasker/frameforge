"""Frozen SwinIR teacher (CLAUDE.md section 4).

Wraps the official, vendored SwinIR classical-SR x2 model
(python/frameforge/models/_swinir_vendor/, Apache-2.0, see THIRD_PARTY.md)
with a fixed, single-frame, RGB-only inference API and a feature-extraction
hook used for feature-distillation (CLAUDE.md 5.3). The teacher is always
frozen: no gradients ever flow into it from this codebase.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from frameforge.models._swinir_vendor.network_swinir import SwinIR

# Params matching the official 001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth
# checkpoint (see scripts/download_pretrained.py), taken verbatim from
# SwinIR's own main_test_swinir.py classical_sr branch.
_SWINIR_X2_KWARGS = dict(
    upscale=2,
    in_chans=3,
    img_size=64,
    window_size=8,
    img_range=1.0,
    depths=[6, 6, 6, 6, 6, 6],
    embed_dim=180,
    num_heads=[6, 6, 6, 6, 6, 6],
    mlp_ratio=2,
    upsampler="pixelshuffle",
    resi_connection="1conv",
)

# SwinIR requires input H, W to be multiples of this (it pads internally in
# the reference test script, not inside forward()); we replicate that here.
WINDOW_SIZE = 8

# Feature-distillation tap point (CLAUDE.md 5.3): the output of
# SwinIR.forward_features() -- i.e. after all six Residual Swin Transformer
# Blocks and the final patch-unembed back into (B, C, H, W) image layout,
# but before conv_after_body + the global residual add. Captured via a
# forward hook on the *top-level* `patch_unembed` module (note: each
# individual RSTB also owns its own internal `patch_unembed`, a different
# module instance used for its own residual conv -- hooking the top-level
# one specifically is what makes this fire exactly once per forward pass,
# with the (B, 180, H, W) image-layout feature forward_features actually
# returns, not an RSTB's internal (B, L, C) token-layout intermediate).
# 180 channels, at LR resolution (SwinIR upsamples only at the very end via
# conv_before_upsample/upsample), matching the resolution the student's
# mid-body feature lives at too.
FEATURE_CHANNELS = 180


class SwinIRTeacher(nn.Module):
    """Frozen wrapper: RGB in, (upscaled RGB, mid-body feature) out."""

    def __init__(self, weights_path: str | Path):
        super().__init__()
        self.net = SwinIR(**_SWINIR_X2_KWARGS)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        state_dict = state["params"] if "params" in state else state
        self.net.load_state_dict(state_dict, strict=True)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

        self._tapped_feature: torch.Tensor | None = None
        # See FEATURE_CHANNELS docstring above for why this exact module
        # (not layers[-1], which is still in token layout).
        self.net.patch_unembed.register_forward_hook(self._capture_feature)

    def _capture_feature(self, module, inputs, output):
        self._tapped_feature = output

    def _pad_to_window_multiple(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        _, _, h, w = x.shape
        pad_h = (WINDOW_SIZE - h % WINDOW_SIZE) % WINDOW_SIZE
        pad_w = (WINDOW_SIZE - w % WINDOW_SIZE) % WINDOW_SIZE
        if pad_h or pad_w:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, pad_h, pad_w

    @torch.no_grad()
    def forward(self, lr_rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            lr_rgb: (B, 3, H, W), tonemapped sRGB in [0, 1] (same
                preprocessing as everywhere else, frameforge.colorspace).

        Returns:
            (sr_rgb, feature): sr_rgb is (B, 3, 2H, 2W) in [0, 1];
            feature is (B, 180, H', W') where H', W' are H, W padded up to
            a multiple of WINDOW_SIZE (the student's feature adapter must
            crop/interpolate to match its own feature map -- see losses.py).
        """
        _, _, h, w = lr_rgb.shape
        padded, pad_h, pad_w = self._pad_to_window_multiple(lr_rgb)
        sr = self.net(padded)
        feature = self._tapped_feature
        # Crop the super-resolved output back to the true (unpadded) size.
        sr = sr[:, :, : h * 2, : w * 2]
        # SwinIR itself does not clamp; the rest of this codebase treats
        # tonemapped sRGB as always in [0, 1] (frameforge.colorspace), so
        # clamp here rather than let out-of-range teacher outputs leak into
        # distillation losses/metrics downstream.
        sr = sr.clamp(0.0, 1.0)
        return sr, feature
