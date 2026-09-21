"""Export FrameForgeNet to ONNX with the warp inside the graph (CLAUDE.md
section 7.2).

IMPORTANT -- "first frame" handling changes shape at the ONNX boundary:
`FrameForgeNet.forward`'s `prev_hr_output=None` convenience (substitute a
bilinear upsample of lr_rgb) is a Python-level branch and cannot be part of
a traced ONNX graph without baking in one fixed behavior. So the exported
graph ALWAYS requires a concrete `prev_hr_output` tensor -- callers
(evaluate.py's ONNX Runtime path, and eventually the C++ engine) must
compute the first frame's substitute themselves (bilinear-upsample lr_rgb)
before running the session, exactly mirroring what `train.run_unroll` does
internally in Python. This keeps the exported graph free of data-dependent
control flow, which ONNX handles poorly anyway.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch
import torch.nn.functional as F

from frameforge.models.student import FrameForgeNet

ONNX_OPSET = 17  # >= 16, required for GridSample (CLAUDE.md 7.2)


def _dummy_inputs(model: FrameForgeNet, h: int = 64, w: int = 96, batch: int = 1):
    lr_rgb = torch.rand(batch, 3, h, w)
    input_names = ["lr_rgb"]
    args = [lr_rgb]
    dynamic_axes = {"lr_rgb": {0: "batch", 2: "height", 3: "width"}}

    if model.use_depth:
        args.append(torch.rand(batch, 1, h, w))
        input_names.append("lr_depth")
        dynamic_axes["lr_depth"] = {0: "batch", 2: "height", 3: "width"}
    if model.use_temporal:
        args.append(torch.zeros(batch, 2, h, w))
        input_names.append("lr_mv")
        dynamic_axes["lr_mv"] = {0: "batch", 2: "height", 3: "width"}

        prev_hr = F.interpolate(
            lr_rgb, scale_factor=model.scale, mode="bilinear", align_corners=False
        )
        args.append(prev_hr)
        input_names.append("prev_hr_output")
        dynamic_axes["prev_hr_output"] = {0: "batch", 2: "height_hr", 3: "width_hr"}

    dynamic_axes["output"] = {0: "batch", 2: "height_hr", 3: "width_hr"}
    return tuple(args), input_names, dynamic_axes


class _ExportWrapper(torch.nn.Module):
    """Forces forward()'s kwargs into positional args matching `_dummy_inputs`'s
    ordering, so torch.onnx.export sees a plain-positional signature (kwargs
    with a None default confuse the tracer's input-name inference)."""

    def __init__(self, model: FrameForgeNet):
        super().__init__()
        self.model = model

    def forward(self, *args):
        idx = 0
        lr_rgb = args[idx]
        idx += 1
        lr_depth = None
        if self.model.use_depth:
            lr_depth = args[idx]
            idx += 1
        lr_mv = None
        prev_hr_output = None
        if self.model.use_temporal:
            lr_mv = args[idx]
            idx += 1
            prev_hr_output = args[idx]
            idx += 1
        return self.model(lr_rgb, lr_depth=lr_depth, lr_mv=lr_mv, prev_hr_output=prev_hr_output)


def export_model(model: FrameForgeNet, out_path: str | Path, h: int = 64, w: int = 96) -> Path:
    model.eval()
    wrapper = _ExportWrapper(model)
    args, input_names, dynamic_axes = _dummy_inputs(model, h=h, w=w)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        args,
        str(out_path),
        input_names=input_names,
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        # Force the legacy TorchScript-based exporter: the newer "dynamo"
        # exporter (torch's default since ~2.5) needs the extra `onnxscript`
        # dependency, which drags in a numpy 2.x upgrade that conflicts with
        # this project's numpy==1.26.4 pin (see pyproject.toml / D3). The
        # legacy exporter handles GridSample at opset>=16 fine and needs no
        # extra dependency.
        dynamo=False,
    )

    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="path to a train.py checkpoint .pt")
    parser.add_argument("--variant", default="temporal")
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model = FrameForgeNet(
            variant=ckpt["variant"], channels=ckpt["channels"], num_blocks=ckpt["num_blocks"]
        )
        model.load_state_dict(ckpt["model_state"])
    else:
        model = FrameForgeNet(
            variant=args.variant, channels=args.channels, num_blocks=args.num_blocks
        )

    out_path = export_model(model, args.out)
    print(f"exported to {out_path}")


if __name__ == "__main__":
    main()
