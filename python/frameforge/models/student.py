"""FrameForgeNet: the distilled student model (CLAUDE.md section 5.1).

The warp (grid_sample) lives INSIDE forward() so the whole recurrent step
-- including motion compensation -- exports as one ONNX graph (opset >= 16
for GridSample support); the C++ engine only ever feeds tensors in and
takes tensors out, never reimplements warping itself.

Recurrence itself (looping over a sequence's frames, feeding each output
back in as the next frame's `prev_hr_output`) is NOT inside this module --
that's the caller's job (train.py's unroll, evaluate.py, and the C++
engine's inference stage, which is explicitly where CLAUDE.md 7.3 says the
recurrent state is held). This module is a pure per-frame function of its
inputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from frameforge.warp import validity_mask, warp

# (use_temporal, use_depth) per CLAUDE.md 5.1's three required variants.
VARIANT_CONFIGS: dict[str, tuple[bool, bool]] = {
    "single_frame": (False, True),       # "no temporal input (depth optional)" -- default depth on
    "single_frame_nodepth": (False, False),
    "temporal": (True, True),
    "temporal_nodepth": (True, False),   # ablation: temporal but no depth
}


class ResidualBlock(nn.Module):
    """conv-ReLU-conv, no batch norm, identity skip (CLAUDE.md 5.1)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x)))


class FrameForgeNet(nn.Module):
    def __init__(
        self,
        variant: str = "temporal",
        channels: int = 32,
        num_blocks: int = 6,
        scale: int = 2,
        use_validity_mask: bool = True,
    ):
        super().__init__()
        if variant not in VARIANT_CONFIGS:
            raise ValueError(f"unknown variant {variant!r}; choose from {list(VARIANT_CONFIGS)}")
        self.variant = variant
        self.use_temporal, self.use_depth = VARIANT_CONFIGS[variant]
        self.use_validity_mask = use_validity_mask and self.use_temporal
        self.scale = scale
        self.channels = channels
        self.num_blocks = num_blocks

        in_channels = 3
        if self.use_depth:
            in_channels += 1
        if self.use_temporal:
            # warped-previous-output, pixel-unshuffled by `scale`: 3 * scale^2
            in_channels += 3 * scale * scale
            if self.use_validity_mask:
                in_channels += 1
        self.in_channels = in_channels

        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.conv_out = nn.Conv2d(channels, 3 * scale * scale, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.pixel_unshuffle = nn.PixelUnshuffle(scale)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        lr_rgb: torch.Tensor,
        lr_depth: torch.Tensor | None = None,
        lr_mv: torch.Tensor | None = None,
        prev_hr_output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One recurrent step.

        Args:
            lr_rgb: (B, 3, H, W), tonemapped sRGB in [0, 1].
            lr_depth: (B, 1, H, W), normalized [0, 1]. Required iff
                `self.use_depth`.
            lr_mv: (B, 2, H, W), backward flow (this frame -> previous
                frame), pixel units AT LR RESOLUTION, (dx, dy)
                (frameforge.warp's convention). Required iff
                `self.use_temporal`. For a sequence's first frame, pass a
                zero tensor (matches frameforge.data.dataset's convention).
            prev_hr_output: (B, 3, H*scale, W*scale), the previous step's
                output O_{t-1}. If None (first frame of a sequence, per
                CLAUDE.md 5.1), substituted with a bilinear upsample of
                `lr_rgb` -- combined with a zero `lr_mv`, warping this is a
                no-op, so the two "first frame" rules are self-consistent.

        Returns:
            (B, 3, H*scale, W*scale) tensor, the upscaled output O_t.
        """
        b, _, h, w = lr_rgb.shape
        features = [lr_rgb]

        if self.use_depth:
            if lr_depth is None:
                raise ValueError(f"variant {self.variant!r} requires lr_depth")
            features.append(lr_depth)

        if self.use_temporal:
            if lr_mv is None:
                raise ValueError(f"variant {self.variant!r} requires lr_mv")
            if prev_hr_output is None:
                prev_hr_output = F.interpolate(
                    lr_rgb, scale_factor=self.scale, mode="bilinear", align_corners=False
                )
            hr_mv = F.interpolate(
                lr_mv, scale_factor=self.scale, mode="bilinear", align_corners=False
            ) * self.scale  # pixel-unit flow scales with resolution
            warped_prev = warp(prev_hr_output, hr_mv)
            warped_prev_lr = self.pixel_unshuffle(warped_prev)  # (B, 3*scale^2, H, W)
            features.append(warped_prev_lr)

            if self.use_validity_mask:
                mask = validity_mask(lr_mv)  # (B, 1, H, W)
                features.append(mask)

        x = torch.cat(features, dim=1)
        x = self.conv_in(x)
        x = self.body(x)
        # Mid-body feature, exposed for feature distillation (CLAUDE.md 5.3).
        # Deliberately just a plain attribute set during forward(), not part
        # of the module graph itself -- the training-only 1x1 adapter that
        # projects this to the teacher's channel count lives in losses.py,
        # so it is trivially excluded from ONNX export (export_onnx.py only
        # traces this module's forward(), never touches `last_feature`).
        self.last_feature = x
        x = self.conv_out(x)
        residual = self.pixel_shuffle(x)

        global_skip = F.interpolate(
            lr_rgb, scale_factor=self.scale, mode="bilinear", align_corners=False
        )
        out = residual + global_skip
        return out.clamp(0.0, 1.0)
