"""FrameForgeNet tests: param budget (enforced, CLAUDE.md M2 acceptance),
shapes across all required variants, first-frame behavior, gradient flow.
"""

from __future__ import annotations

import torch

from frameforge.models.student import VARIANT_CONFIGS, FrameForgeNet

PARAM_BUDGET = 300_000


def test_param_budget_enforced_for_all_variants_default_config():
    """CLAUDE.md 5.2 M2 acceptance: "param budget enforced by a unit test"."""
    for variant in VARIANT_CONFIGS:
        model = FrameForgeNet(variant=variant, channels=32, num_blocks=6)
        n_params = model.num_parameters()
        assert n_params <= PARAM_BUDGET, (
            f"variant {variant!r} has {n_params} params, over the {PARAM_BUDGET} budget"
        )


def test_forward_shapes_all_variants():
    b, h, w = 2, 16, 24
    lr_rgb = torch.rand(b, 3, h, w)
    lr_depth = torch.rand(b, 1, h, w)
    lr_mv = torch.zeros(b, 2, h, w)
    prev = torch.rand(b, 3, h * 2, w * 2)

    for variant, (use_temporal, use_depth) in VARIANT_CONFIGS.items():
        model = FrameForgeNet(variant=variant, channels=16, num_blocks=2).eval()
        kwargs = {}
        if use_depth:
            kwargs["lr_depth"] = lr_depth
        if use_temporal:
            kwargs["lr_mv"] = lr_mv
            kwargs["prev_hr_output"] = prev
        with torch.no_grad():
            out = model(lr_rgb, **kwargs)
        assert out.shape == (b, 3, h * 2, w * 2), f"variant {variant} wrong output shape"
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_first_frame_with_no_prev_output_matches_bilinear_when_body_is_zeroed():
    """With zero mv and no prev_hr_output, the model substitutes a bilinear
    upsample of lr_rgb as prev, then warps it with a zero flow (a no-op) --
    so if we also zero the body's contribution, output == bilinear upsample
    exactly. This isolates and checks the "first frame" plumbing itself,
    independent of learned weights."""
    b, h, w = 1, 12, 12
    model = FrameForgeNet(variant="temporal", channels=8, num_blocks=1).eval()
    with torch.no_grad():
        for p in model.conv_out.parameters():
            p.zero_()  # residual branch contributes exactly zero

    lr_rgb = torch.rand(b, 3, h, w)
    lr_depth = torch.rand(b, 1, h, w)
    lr_mv = torch.zeros(b, 2, h, w)

    with torch.no_grad():
        out = model(lr_rgb, lr_depth=lr_depth, lr_mv=lr_mv, prev_hr_output=None)

    import torch.nn.functional as F
    expected = F.interpolate(lr_rgb, scale_factor=2, mode="bilinear", align_corners=False)
    assert torch.allclose(out, expected.clamp(0, 1), atol=1e-5)


def test_gradients_flow_to_all_parameters():
    b, h, w = 1, 16, 16
    model = FrameForgeNet(variant="temporal", channels=8, num_blocks=2)
    lr_rgb = torch.rand(b, 3, h, w, requires_grad=False)
    lr_depth = torch.rand(b, 1, h, w)
    lr_mv = torch.zeros(b, 2, h, w)
    prev = torch.rand(b, 3, h * 2, w * 2)

    out = model(lr_rgb, lr_depth=lr_depth, lr_mv=lr_mv, prev_hr_output=prev)
    loss = out.mean()
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient at {name}"


def test_single_frame_variant_ignores_temporal_args():
    model = FrameForgeNet(variant="single_frame", channels=8, num_blocks=1).eval()
    lr_rgb = torch.rand(1, 3, 12, 12)
    lr_depth = torch.rand(1, 1, 12, 12)
    with torch.no_grad():
        out = model(lr_rgb, lr_depth=lr_depth)  # no lr_mv/prev needed
    assert out.shape == (1, 3, 24, 24)
