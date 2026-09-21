"""Evaluate every baseline and student variant on a held-out test set
(CLAUDE.md section 6), writing per-(model, sequence) rows to a CSV.

`scripts/gen_results_table.py` aggregates the CSV this writes into
`docs/RESULTS.md`'s markdown table -- this script owns computing numbers,
that one owns formatting them, per CLAUDE.md 1's "every metric... produced
by a committed script" rule (no numbers get typed into RESULTS.md by hand).
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from frameforge.data.dataset import FrameSequenceDataset
from frameforge.metrics import (
    LPIPSMetric,
    compute_all_metrics,
    count_parameters,
    estimate_gflops,
)
from frameforge.models.student import FrameForgeNet
from frameforge.models.teacher import SwinIRTeacher
from frameforge.train import run_unroll


def bilinear_baseline(sample) -> torch.Tensor:
    return F.interpolate(sample.lr_rgb, scale_factor=2, mode="bilinear", align_corners=False)


def bicubic_baseline(sample) -> torch.Tensor:
    return F.interpolate(sample.lr_rgb, scale_factor=2, mode="bicubic", align_corners=False)


def fsr1_baseline(sample, fsr1_cli_path: str, sharpness: float | None = 0.2) -> torch.Tensor:
    """Runs the compiled FSR1 CPU port via subprocess, one frame at a time
    (CLAUDE.md 6.1: "a subprocess for evaluation"), round-tripping through
    PNG since that's the CLI's I/O format.
    """
    t, _, h, w = sample.lr_rgb.shape
    out_h, out_w = h * 2, w * 2
    outputs = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for frame_idx in range(t):
            frame = sample.lr_rgb[frame_idx].clamp(0, 1)
            arr = (frame.permute(1, 2, 0) * 255).round().byte().numpy()
            in_path = tmp_path / f"in_{frame_idx}.png"
            out_path = tmp_path / f"out_{frame_idx}.png"
            Image.fromarray(arr).save(in_path)

            cmd = [fsr1_cli_path, "--in", str(in_path), "--out", str(out_path),
                   "--width", str(out_w), "--height", str(out_h)]
            if sharpness is not None:
                cmd += ["--sharpness", str(sharpness)]
            subprocess.run(cmd, check=True, capture_output=True)

            out_img = Image.open(out_path).convert("RGB")
            out_tensor = torch.from_numpy(np.array(out_img)).permute(2, 0, 1).float() / 255.0
            outputs.append(out_tensor)
    return torch.stack(outputs, dim=0)


def student_prediction(model: FrameForgeNet, sample) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return run_unroll(model, sample)


def measure_onnx_latency_ms(onnx_path: str, h: int, w: int, model: FrameForgeNet,
                             n_warmup: int = 3, n_iters: int = 20) -> float:
    """Median ms/frame via ONNX Runtime CPU EP (CLAUDE.md 6: "ONNX Runtime
    latency"). A separate, C++-engine-measured latency (real pipeline,
    threading, tiling) is reported in RESULTS.md from M5's benchmarks --
    this is the simpler "just run the exported graph" number.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inputs = {"lr_rgb": np.random.rand(1, 3, h, w).astype(np.float32)}
    if model.use_depth:
        inputs["lr_depth"] = np.random.rand(1, 1, h, w).astype(np.float32)
    if model.use_temporal:
        inputs["lr_mv"] = np.zeros((1, 2, h, w), dtype=np.float32)
        inputs["prev_hr_output"] = np.random.rand(1, 3, h * 2, w * 2).astype(np.float32)

    for _ in range(n_warmup):
        session.run(None, inputs)

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        session.run(None, inputs)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def evaluate_model(
    name: str,
    predict_fn,
    dataset: FrameSequenceDataset,
    out_rows: list[dict],
    lpips_metric: LPIPSMetric | None,
    extra_fields: dict | None = None,
):
    for i in range(len(dataset)):
        sample = dataset[i]
        pred = predict_fn(sample)
        metrics = compute_all_metrics(
            pred, sample.hr_rgb, lr_mv=sample.lr_mv, lpips_metric=lpips_metric
        )
        row = {"model": name, "scene": sample.scene, "sequence": sample.sequence, **metrics}
        if extra_fields:
            row.update(extra_fields)
        out_rows.append(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--test-scenes", nargs="*", default=None,
                         help="defaults to configs/data.yaml's sequences.test_scenes")
    parser.add_argument("--teacher-weights", default=None)
    parser.add_argument("--fsr1-cli", default=None,
                         help="path to the built fsr1_cli(.exe) binary; FSR1 baseline "
                              "skipped if not given")
    parser.add_argument("--student-checkpoints", nargs="*", default=[],
                         help="name=path.pt pairs, e.g. "
                              "C_output_feature_kd=checkpoints/C/latest.pt")
    parser.add_argument("--skip-lpips", action="store_true",
                         help="skip LPIPS (avoids needing its pretrained weights)")
    parser.add_argument("--out", default="results/metrics.csv")
    args = parser.parse_args()

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    test_scenes = args.test_scenes or data_cfg["sequences"].get("test_scenes") or None

    dataset = FrameSequenceDataset(
        args.data_root, data_cfg["exr_channel_map"], data_cfg["tonemap"], scenes=test_scenes,
    )

    lpips_metric = None if args.skip_lpips else LPIPSMetric()
    rows: list[dict] = []

    evaluate_model("bilinear", bilinear_baseline, dataset, rows, lpips_metric)
    evaluate_model("bicubic", bicubic_baseline, dataset, rows, lpips_metric)

    if args.fsr1_cli:
        evaluate_model(
            "fsr1_easu_rcas",
            lambda s: fsr1_baseline(s, args.fsr1_cli),
            dataset, rows, lpips_metric,
        )
    else:
        print("--fsr1-cli not given; skipping FSR1 baseline")

    if args.teacher_weights:
        teacher = SwinIRTeacher(args.teacher_weights)

        def teacher_predict(sample):
            out, _ = teacher(sample.lr_rgb)
            return out

        evaluate_model(
            "swinir_teacher", teacher_predict, dataset, rows, lpips_metric,
            extra_fields={"params": count_parameters(teacher)},
        )
    else:
        print("--teacher-weights not given; skipping SwinIR teacher")

    for spec in args.student_checkpoints:
        name, ckpt_path = spec.split("=", 1)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model = FrameForgeNet(
            variant=ckpt["variant"], channels=ckpt["channels"], num_blocks=ckpt["num_blocks"]
        )
        model.load_state_dict(ckpt["model_state"])
        extra = {"params": count_parameters(model), "gflops_540p": estimate_gflops(model)}
        evaluate_model(name, lambda s, m=model: student_prediction(m, s), dataset, rows,
                        lpips_metric, extra_fields=extra)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row})
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
