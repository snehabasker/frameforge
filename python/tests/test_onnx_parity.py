"""ONNX export + PyTorch/ONNX-Runtime parity test (CLAUDE.md section 7.2:
"max abs error < 1e-4 on fixed inputs. This test must pass in CI.").

Uses a tiny model (few thousand params) so this is fast and needs no
downloaded weights -- runs in the `python` CI job.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest
import torch

from frameforge.export_onnx import export_model
from frameforge.models.student import VARIANT_CONFIGS, FrameForgeNet


@pytest.mark.parametrize("variant", list(VARIANT_CONFIGS))
def test_onnx_parity_max_abs_error(variant, tmp_path):
    torch.manual_seed(0)
    model = FrameForgeNet(variant=variant, channels=8, num_blocks=2).eval()

    h, w = 20, 24  # deliberately not a "nice" power-of-two size
    out_path = export_model(model, tmp_path / f"{variant}.onnx", h=h, w=w)

    # Build the SAME fixed inputs export_model used internally, but with a
    # fresh random seed so this isn't just re-checking export's own dummy
    # tensors -- a real, independent input.
    torch.manual_seed(123)
    lr_rgb = torch.rand(1, 3, h, w)
    torch_kwargs = {}
    onnx_inputs = {"lr_rgb": lr_rgb.numpy()}

    if model.use_depth:
        lr_depth = torch.rand(1, 1, h, w)
        torch_kwargs["lr_depth"] = lr_depth
        onnx_inputs["lr_depth"] = lr_depth.numpy()
    if model.use_temporal:
        lr_mv = torch.randn(1, 2, h, w) * 2.0
        prev_hr = torch.rand(1, 3, h * model.scale, w * model.scale)
        torch_kwargs["lr_mv"] = lr_mv
        torch_kwargs["prev_hr_output"] = prev_hr
        onnx_inputs["lr_mv"] = lr_mv.numpy()
        onnx_inputs["prev_hr_output"] = prev_hr.numpy()

    with torch.no_grad():
        torch_out = model(lr_rgb, **torch_kwargs).numpy()

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, onnx_inputs)[0]

    max_abs_error = np.max(np.abs(torch_out - onnx_out))
    assert max_abs_error < 1e-4, f"variant {variant}: max abs error {max_abs_error} >= 1e-4"


@pytest.mark.parametrize("variant", list(VARIANT_CONFIGS))
def test_onnx_export_supports_dynamic_hw(variant, tmp_path):
    """The graph must accept a DIFFERENT H/W than it was exported with
    (CLAUDE.md 7.2: "dynamic H/W axes") -- exercised here at half the
    export-time resolution."""
    torch.manual_seed(0)
    model = FrameForgeNet(variant=variant, channels=8, num_blocks=2).eval()
    out_path = export_model(model, tmp_path / f"{variant}.onnx", h=20, w=24)

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    h, w = 12, 16
    inputs = {"lr_rgb": np.random.rand(1, 3, h, w).astype(np.float32)}
    if model.use_depth:
        inputs["lr_depth"] = np.random.rand(1, 1, h, w).astype(np.float32)
    if model.use_temporal:
        inputs["lr_mv"] = np.zeros((1, 2, h, w), dtype=np.float32)
        inputs["prev_hr_output"] = np.random.rand(1, 3, h * model.scale, w * model.scale).astype(
            np.float32
        )
    out = session.run(None, inputs)[0]
    assert out.shape == (1, 3, h * model.scale, w * model.scale)
