"""DistillationLosses tests: each term activates only when its weight > 0,
and the four experiment-matrix presets match CLAUDE.md's table."""

from __future__ import annotations

import pytest
import torch

from frameforge.losses import EXPERIMENT_PRESETS, DistillationLosses, LossWeights


def _dummy_tensors(t=3, h=8, w=8, scale=2, student_ch=16, teacher_ch=180):
    student_out = torch.rand(t, 3, h * scale, w * scale)
    hr_target = torch.rand(t, 3, h * scale, w * scale)
    teacher_out = torch.rand(t, 3, h * scale, w * scale)
    teacher_feat = torch.rand(t, teacher_ch, h, w)
    student_feat = torch.rand(t, student_ch, h, w)
    lr_mv = torch.zeros(t, 2, h, w)
    return student_out, hr_target, teacher_out, teacher_feat, student_feat, lr_mv


def test_recon_only_by_default():
    losses = DistillationLosses(LossWeights())
    student_out, hr_target, *_ = _dummy_tensors()
    out = losses(student_out, hr_target)
    assert set(out.keys()) == {"recon", "total"}
    assert torch.isclose(out["total"], out["recon"])


def test_output_kd_requires_teacher_out():
    losses = DistillationLosses(LossWeights(output_kd=0.5))
    student_out, hr_target, *_ = _dummy_tensors()
    with pytest.raises(ValueError):
        losses(student_out, hr_target)


def test_all_four_terms_active():
    weights = LossWeights(recon=1.0, output_kd=0.5, feature_kd=0.1, temporal=0.1)
    losses = DistillationLosses(weights, student_channels=16, teacher_channels=180)
    student_out, hr_target, teacher_out, teacher_feat, student_feat, lr_mv = _dummy_tensors()
    out = losses(
        student_out, hr_target,
        teacher_out=teacher_out, teacher_feat=teacher_feat,
        student_feat=student_feat, lr_mv=lr_mv,
    )
    assert set(out.keys()) == {"recon", "output_kd", "feature_kd", "temporal", "total"}
    for k, v in out.items():
        assert torch.isfinite(v), f"{k} is not finite"


def test_feature_kd_adapter_only_created_when_needed():
    losses_on = DistillationLosses(LossWeights(feature_kd=0.1))
    losses_off = DistillationLosses(LossWeights(feature_kd=0.0))
    assert losses_on.adapter is not None
    assert losses_off.adapter is None


def test_experiment_presets_match_spec_table():
    # CLAUDE.md 5.3 matrix: A=recon+temporal, B=+output KD, C=+feature KD,
    # D=C without temporal.
    a, b, c, d = (
        EXPERIMENT_PRESETS["A_scratch"],
        EXPERIMENT_PRESETS["B_output_kd"],
        EXPERIMENT_PRESETS["C_output_feature_kd"],
        EXPERIMENT_PRESETS["D_no_temporal"],
    )
    assert a.output_kd == 0.0 and a.feature_kd == 0.0 and a.temporal > 0.0
    assert b.output_kd > 0.0 and b.feature_kd == 0.0 and b.temporal > 0.0
    assert c.output_kd > 0.0 and c.feature_kd > 0.0 and c.temporal > 0.0
    assert d.output_kd == c.output_kd and d.feature_kd == c.feature_kd and d.temporal == 0.0


def test_temporal_loss_zero_for_static_perfectly_matching_sequence():
    """If every frame is identical and flow is zero, warp(prev)==prev==cur,
    so the temporal loss should be ~0."""
    losses = DistillationLosses(LossWeights(temporal=1.0))
    t, h, w = 3, 8, 8
    frame = torch.rand(1, 3, h * 2, w * 2)
    student_out = frame.repeat(t, 1, 1, 1)
    hr_target = student_out.clone()
    lr_mv = torch.zeros(t, 2, h, w)
    out = losses(student_out, hr_target, lr_mv=lr_mv)
    assert out["temporal"].item() < 1e-5
