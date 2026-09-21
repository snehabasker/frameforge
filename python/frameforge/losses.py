"""Reconstruction, distillation, and temporal-consistency losses
(CLAUDE.md section 5.3).

The four losses map onto the spec's experiment matrix (A: recon+temporal,
B: +output KD, C: +feature KD, D: C without temporal) purely through which
`LossWeights` are non-zero -- `DistillationLosses` always computes whichever
terms have a non-zero weight and skips the rest, so one class serves all
four runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from frameforge.warp import validity_mask, warp


@dataclass
class LossWeights:
    recon: float = 1.0
    output_kd: float = 0.0
    feature_kd: float = 0.0
    temporal: float = 0.0


# Experiment matrix presets (CLAUDE.md 5.3 table), for configs/train_*.yaml
# to reference by name instead of repeating the four numbers everywhere.
EXPERIMENT_PRESETS: dict[str, LossWeights] = {
    "A_scratch": LossWeights(recon=1.0, output_kd=0.0, feature_kd=0.0, temporal=0.1),
    "B_output_kd": LossWeights(recon=1.0, output_kd=0.5, feature_kd=0.0, temporal=0.1),
    "C_output_feature_kd": LossWeights(recon=1.0, output_kd=0.5, feature_kd=0.1, temporal=0.1),
    "D_no_temporal": LossWeights(recon=1.0, output_kd=0.5, feature_kd=0.1, temporal=0.0),
}


class DistillationLosses(nn.Module):
    def __init__(
        self,
        weights: LossWeights,
        student_channels: int = 32,
        teacher_channels: int = 180,
    ):
        super().__init__()
        self.weights = weights
        # 1x1 adapter, training-only: projects the student's mid-body
        # feature to the teacher's channel count for feature distillation.
        # Never touched by export_onnx.py (that only traces FrameForgeNet).
        self.adapter = (
            nn.Conv2d(student_channels, teacher_channels, kernel_size=1)
            if weights.feature_kd > 0
            else None
        )

    def forward(
        self,
        student_out: torch.Tensor,
        hr_target: torch.Tensor,
        teacher_out: torch.Tensor | None = None,
        teacher_feat: torch.Tensor | None = None,
        student_feat: torch.Tensor | None = None,
        lr_mv: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """All tensors are (T, C, H, W) (batch dim folded into T, as
        `train.run_unroll` produces). Returns a dict of scalar losses plus
        "total"; unused terms are omitted from the dict entirely (not
        zero-valued) so callers/logging can tell what actually ran.
        """
        out: dict[str, torch.Tensor] = {}
        out["recon"] = F.l1_loss(student_out, hr_target)
        total = self.weights.recon * out["recon"]

        if self.weights.output_kd > 0:
            if teacher_out is None:
                raise ValueError("output_kd weight > 0 but teacher_out is None")
            out["output_kd"] = F.l1_loss(student_out, teacher_out)
            total = total + self.weights.output_kd * out["output_kd"]

        if self.weights.feature_kd > 0:
            if teacher_feat is None or student_feat is None:
                raise ValueError("feature_kd weight > 0 but teacher_feat/student_feat is None")
            projected = self.adapter(student_feat)
            if projected.shape[-2:] != teacher_feat.shape[-2:]:
                projected = F.interpolate(
                    projected, size=teacher_feat.shape[-2:], mode="bilinear", align_corners=False
                )
            out["feature_kd"] = F.mse_loss(projected, teacher_feat)
            total = total + self.weights.feature_kd * out["feature_kd"]

        if self.weights.temporal > 0:
            if lr_mv is None:
                raise ValueError("temporal weight > 0 but lr_mv is None")
            t = student_out.shape[0]
            if t > 1:
                scale = student_out.shape[-1] // lr_mv.shape[-1]
                hr_mv = F.interpolate(
                    lr_mv[1:], scale_factor=scale, mode="bilinear", align_corners=False
                ) * scale
                warped_prev = warp(student_out[:-1], hr_mv)
                mask = validity_mask(lr_mv[1:])  # (T-1, 1, H_lr, W_lr)
                mask_hr = F.interpolate(mask, scale_factor=scale, mode="nearest")
                diff = (student_out[1:] - warped_prev).abs() * mask_hr
                denom = mask_hr.sum().clamp(min=1.0) * student_out.shape[1]
                out["temporal"] = diff.sum() / denom
                total = total + self.weights.temporal * out["temporal"]

        out["total"] = total
        return out


def ground_truth_temporal_error(hr_rgb: torch.Tensor, lr_mv: torch.Tensor) -> torch.Tensor:
    """The SAME temporal-error quantity computed on ground truth instead of
    student output (CLAUDE.md 6: "reported alongside the same quantity for
    ground truth -- the gap is what matters"). Used by evaluate.py.
    """
    t = hr_rgb.shape[0]
    if t <= 1:
        return torch.tensor(0.0)
    scale = hr_rgb.shape[-1] // lr_mv.shape[-1]
    hr_mv = F.interpolate(
        lr_mv[1:], scale_factor=scale, mode="bilinear", align_corners=False
    ) * scale
    warped_prev = warp(hr_rgb[:-1], hr_mv)
    mask = validity_mask(lr_mv[1:])
    mask_hr = F.interpolate(mask, scale_factor=scale, mode="nearest")
    diff = (hr_rgb[1:] - warped_prev).abs() * mask_hr
    denom = mask_hr.sum().clamp(min=1.0) * hr_rgb.shape[1]
    return diff.sum() / denom
