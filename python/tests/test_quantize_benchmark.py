"""scripts/quantize_benchmark.py tests: tiny model, synthetic calibration, no
data or checkpoints needed (runs in the `python` CI job)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnxruntime.quantization import QuantFormat

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from quantize_benchmark import (  # noqa: E402
    RecurrentCalibrationReader,
    benchmark_latency,
    quantize,
    run_recurrent,
)

from frameforge.export_onnx import export_model  # noqa: E402
from frameforge.metrics import psnr  # noqa: E402
from frameforge.models.student import FrameForgeNet  # noqa: E402


class _FakeDataset:
    def __init__(self, n_seq: int, frames: int, h: int, w: int):
        g = torch.Generator().manual_seed(0)
        self._samples = [
            SimpleNamespace(
                lr_rgb=torch.rand(frames, 3, h, w, generator=g),
                lr_depth=torch.rand(frames, 1, h, w, generator=g),
                lr_mv=torch.zeros(frames, 2, h, w),
            )
            for _ in range(n_seq)
        ]

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, i):
        return self._samples[i]


@pytest.fixture(scope="module")
def quantized(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("quant")
    torch.manual_seed(0)
    model = FrameForgeNet(variant="temporal", channels=16, num_blocks=2).eval()
    fp32 = export_model(model, tmp / "fp32.onnx", h=16, w=24)
    dataset = _FakeDataset(n_seq=2, frames=3, h=16, w=24)
    reader = RecurrentCalibrationReader(fp32, model, dataset)
    int8 = quantize(fp32, tmp / "int8.onnx", reader, QuantFormat.QDQ)
    return model, dataset, fp32, int8, reader


def test_calibration_reader_yields_every_frame_then_rewinds(quantized):
    *_, reader = quantized
    assert len(reader) == 6
    reader.rewind()  # the fixture's quantize() call already drained it
    count = 0
    while reader.get_next() is not None:
        count += 1
    assert count == 6
    reader.rewind()
    assert reader.get_next() is not None


def test_int8_graph_is_actually_quantized(quantized):
    import onnx

    _, _, _, int8, _ = quantized
    ops = {n.op_type for n in onnx.load(str(int8)).graph.node}
    assert "QuantizeLinear" in ops and "DequantizeLinear" in ops


def test_int8_output_close_to_fp32(quantized):
    model, dataset, fp32, int8, _ = quantized
    sample = dataset[0]
    ref = run_recurrent(ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"]),
                        model, sample)
    out = run_recurrent(ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"]),
                        model, sample)
    assert out.shape == ref.shape
    # Loose floor: a tiny random-weight net quantizes crudely, this only
    # guards against a broken graph, not a quality regression.
    assert psnr(out, ref).item() > 25.0


def test_benchmark_latency_returns_ordered_positive_stats(quantized):
    model, _, fp32, _, _ = quantized
    median, p95 = benchmark_latency(fp32, model, h=16, w=24, threads=1, warmup=1, iters=5)
    assert 0 < median <= p95
    assert np.isfinite(p95)


def test_render_markdown_uses_only_csv_values():
    from quantize_benchmark import render_markdown

    def row(variant, size, med):
        return {
            "variant": variant, "size_kb": size, "size_ratio": "1.0", "median_ms_t1": med,
            "p95_ms_t1": "9", "speedup_t1": "1.0", "psnr_vs_fp32_db": "43.1",
            "psnr_vs_gt_db": "16.9", "checkpoint": "ck.pt", "input_lr": "960x540",
            "cpu": "TestCPU", "logical_cores": "4", "os": "TestOS", "onnxruntime": "1.2.3",
            "torch": "4.5.6", "calibration_frames": "3",
        }

    md = render_markdown([row("fp32", "493.2", "111.1"), row("int8_qdq", "213.6", "22.2")],
                         Path("results/quantization.csv"))
    for expected in ("493.2", "213.6", "111.1", "22.2", "TestCPU", "1.2.3", "smoke test"):
        assert expected in md
    assert md.count("| fp32 |") == 1
