"""Export FrameForgeNet to ONNX, quantize it to INT8 with ONNX Runtime, and
benchmark size, CPU latency and output fidelity against the FP32 graph.

    python scripts/quantize_benchmark.py --checkpoint runs/demo_checkpoints/C/latest.pt

Variants produced (all static post-training quantization -- dynamic
quantization only covers MatMul/Gemm-style ops, not this conv-only body):

    fp32          the exported graph, unchanged
    int8_qdq      QDQ format, per-channel S8 weights, U8 activations
    int8_qop      QOperator format (QLinearConv), same bit widths

Calibration activations come from running the FP32 model recurrently over
real frame sequences (so `prev_hr_output` has realistic statistics), not from
random noise. Fidelity is measured by running each ONNX graph recurrently on
the same sequences: PSNR(variant, fp32) says how far quantization moved the
output, PSNR(variant, HR ground truth) says what it cost in actual quality.

Latency is a plain `session.run` timing at LR (H, W) -> (2H, 2W), CPU EP --
the same kind of number as `evaluate.measure_onnx_latency_ms`, not the C++
engine's pipeline throughput.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

from frameforge.data.dataset import FrameSequenceDataset
from frameforge.export_onnx import export_model
from frameforge.metrics import psnr
from frameforge.models.student import FrameForgeNet


def load_model(checkpoint: str | None, variant: str, channels: int, num_blocks: int):
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = FrameForgeNet(
            variant=ckpt["variant"], channels=ckpt["channels"], num_blocks=ckpt["num_blocks"]
        )
        model.load_state_dict(ckpt["model_state"])
    else:
        model = FrameForgeNet(variant=variant, channels=channels, num_blocks=num_blocks)
    return model.eval()


def frame_inputs(model: FrameForgeNet, sample, t: int, prev_hr: np.ndarray) -> dict:
    inputs = {"lr_rgb": sample.lr_rgb[t : t + 1].numpy()}
    if model.use_depth:
        inputs["lr_depth"] = sample.lr_depth[t : t + 1].numpy()
    if model.use_temporal:
        inputs["lr_mv"] = sample.lr_mv[t : t + 1].numpy()
        inputs["prev_hr_output"] = prev_hr
    return inputs


def first_prev_hr(sample, scale: int) -> np.ndarray:
    """The ONNX graph can't substitute a missing previous output itself (see
    export_onnx.py), so the caller supplies the bilinear upsample of frame 0."""
    up = torch.nn.functional.interpolate(
        sample.lr_rgb[0:1], scale_factor=scale, mode="bilinear", align_corners=False
    )
    return up.numpy()


def run_recurrent(session: ort.InferenceSession, model: FrameForgeNet, sample) -> torch.Tensor:
    outputs = []
    prev = first_prev_hr(sample, model.scale)
    for t in range(sample.lr_rgb.shape[0]):
        (out,) = session.run(None, frame_inputs(model, sample, t, prev))
        outputs.append(torch.from_numpy(out[0]))
        prev = out
    return torch.stack(outputs)


class RecurrentCalibrationReader(CalibrationDataReader):
    """Feeds every frame of every sequence; `prev_hr_output` comes from the
    FP32 graph's own recurrence so the calibrated ranges match deployment."""

    def __init__(self, fp32_path: Path, model: FrameForgeNet, dataset: FrameSequenceDataset):
        session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
        self._feeds: list[dict] = []
        for i in range(len(dataset)):
            sample = dataset[i]
            prev = first_prev_hr(sample, model.scale)
            for t in range(sample.lr_rgb.shape[0]):
                feed = frame_inputs(model, sample, t, prev)
                self._feeds.append(feed)
                (prev,) = session.run(None, feed)
        self._iter = iter(self._feeds)

    def __len__(self) -> int:
        return len(self._feeds)

    def get_next(self):
        return next(self._iter, None)

    def rewind(self):
        self._iter = iter(self._feeds)


def quantize(fp32_path: Path, out_path: Path, reader, quant_format: QuantFormat) -> Path:
    quantize_static(
        str(fp32_path),
        str(out_path),
        reader,
        quant_format=quant_format,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    onnx.checker.check_model(onnx.load(str(out_path)))
    return out_path


def op_histogram(path: Path) -> dict[str, int]:
    hist: dict[str, int] = {}
    for node in onnx.load(str(path)).graph.node:
        hist[node.op_type] = hist.get(node.op_type, 0) + 1
    return hist


def benchmark_latency(path: Path, model: FrameForgeNet, h: int, w: int, threads: int,
                      warmup: int, iters: int) -> tuple[float, float]:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    inputs = {"lr_rgb": rng.random((1, 3, h, w), dtype=np.float32)}
    if model.use_depth:
        inputs["lr_depth"] = rng.random((1, 1, h, w), dtype=np.float32)
    if model.use_temporal:
        inputs["lr_mv"] = np.zeros((1, 2, h, w), dtype=np.float32)
        inputs["prev_hr_output"] = rng.random(
            (1, 3, h * model.scale, w * model.scale), dtype=np.float32
        )

    for _ in range(warmup):
        session.run(None, inputs)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        session.run(None, inputs)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    p95 = times[min(len(times) - 1, int(round(0.95 * (len(times) - 1))))]
    return statistics.median(times), p95


def cpu_name() -> str:
    return platform.processor() or platform.machine()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=None,
                        help="train.py checkpoint; omitted -> randomly initialised model "
                             "(latency/size only, fidelity numbers are then meaningless)")
    parser.add_argument("--variant", default="temporal")
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--data-root", default="tests/fixtures/sequences")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--work-dir", default="runs/quantization")
    parser.add_argument("--height", type=int, default=540, help="LR height for the latency run")
    parser.add_argument("--width", type=int, default=960, help="LR width for the latency run")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, os.cpu_count() or 1])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--out-csv", default="results/quantization.csv")
    args = parser.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, args.variant, args.channels, args.num_blocks)
    fp32_path = export_model(model, work / "fp32.onnx")

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    dataset = FrameSequenceDataset(args.data_root, data_cfg["exr_channel_map"],
                                   data_cfg["tonemap"])
    reader = RecurrentCalibrationReader(fp32_path, model, dataset)
    print(f"calibrating on {len(reader)} frames from {len(dataset)} sequence(s)")

    paths = {"fp32": fp32_path}
    for name, fmt in (("int8_qdq", QuantFormat.QDQ), ("int8_qop", QuantFormat.QOperator)):
        reader.rewind()
        paths[name] = quantize(fp32_path, work / f"{name}.onnx", reader, fmt)

    # Fidelity: recurrent run of every graph over every sequence.
    samples = [dataset[i] for i in range(len(dataset))]
    fp32_session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    fp32_outs = [run_recurrent(fp32_session, model, s) for s in samples]

    rows = []
    for name, path in paths.items():
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        vs_fp32, vs_gt = [], []
        for sample, ref in zip(samples, fp32_outs, strict=True):
            out = run_recurrent(session, model, sample)
            vs_fp32.append(psnr(out, ref).item())
            vs_gt.append(psnr(out, sample.hr_rgb).item())

        row = {
            "variant": name,
            "size_kb": round(path.stat().st_size / 1024, 1),
            "psnr_vs_fp32_db": round(statistics.mean(vs_fp32), 2),
            "psnr_vs_gt_db": round(statistics.mean(vs_gt), 2),
        }
        for n_threads in args.threads:
            med, p95 = benchmark_latency(path, model, args.height, args.width, n_threads,
                                         args.warmup, args.iters)
            row[f"median_ms_t{n_threads}"] = round(med, 1)
            row[f"p95_ms_t{n_threads}"] = round(p95, 1)
        rows.append(row)
        print(row, "ops:", op_histogram(path))

    base = rows[0]
    for row in rows:
        for n_threads in args.threads:
            key = f"median_ms_t{n_threads}"
            row[f"speedup_t{n_threads}"] = round(base[key] / row[key], 2)
        row["size_ratio"] = round(base["size_kb"] / row["size_kb"], 2)
        row.update(
            checkpoint=args.checkpoint or "random-init", input_lr=f"{args.width}x{args.height}",
            calibration_frames=len(reader), cpu=cpu_name(), logical_cores=os.cpu_count(),
            onnxruntime=ort.__version__, torch=torch.__version__, os=platform.platform(),
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
