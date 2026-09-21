"""evaluate.py tests: baseline predictors, and an end-to-end smoke run
against the fixture producing a real CSV (mirrors what a real test-scene
evaluation run would do, on synthetic data -- see docs/DECISIONS.md D0).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
FSR1_CLI = REPO_ROOT / "engine" / "build" / "fsr1_easu" / "fsr1_cli.exe"


def _cfg():
    with open(REPO_ROOT / "configs" / "data.yaml") as f:
        return yaml.safe_load(f)


def _dataset():
    from frameforge.data.dataset import FrameSequenceDataset

    cfg = _cfg()
    return FrameSequenceDataset(FIXTURES / "sequences", cfg["exr_channel_map"], cfg["tonemap"])


def test_bilinear_and_bicubic_baselines_shapes():
    from frameforge.evaluate import bicubic_baseline, bilinear_baseline

    sample = _dataset()[0]
    bl = bilinear_baseline(sample)
    bc = bicubic_baseline(sample)
    assert bl.shape == sample.hr_rgb.shape
    assert bc.shape == sample.hr_rgb.shape
    assert not torch.allclose(bl, bc)  # different interpolation kernels


def test_evaluate_model_writes_expected_row_fields():
    from frameforge.evaluate import bilinear_baseline, evaluate_model

    dataset = _dataset()
    rows = []
    evaluate_model("bilinear", bilinear_baseline, dataset, rows, lpips_metric=None)
    assert len(rows) == 1
    row = rows[0]
    assert row["model"] == "bilinear"
    assert row["scene"] == "synthetic_pan_00"
    for key in ["psnr_rgb", "ssim_rgb", "psnr_y", "ssim_y",
                "temporal_error_pred", "temporal_error_gt"]:
        assert key in row


@pytest.mark.skipif(not FSR1_CLI.exists(), reason="fsr1_cli not built (needs the C++ toolchain)")
def test_fsr1_baseline_via_subprocess():
    from frameforge.evaluate import fsr1_baseline

    sample = _dataset()[0]
    out = fsr1_baseline(sample, str(FSR1_CLI))
    assert out.shape == sample.hr_rgb.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_full_evaluate_cli_smoke(tmp_path):
    """Runs evaluate.py's main() end to end (bilinear/bicubic always, FSR1
    and teacher if available) and checks a real CSV comes out."""
    import subprocess
    import sys

    out_csv = tmp_path / "metrics.csv"
    cmd = [
        sys.executable, "-m", "frameforge.evaluate",
        "--data-root", str(FIXTURES / "sequences"),
        "--skip-lpips",
        "--out", str(out_csv),
    ]
    env_pythonpath = str(REPO_ROOT / "python")
    result = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": env_pythonpath},
    )
    assert result.returncode == 0, result.stderr
    assert out_csv.exists()
    content = out_csv.read_text()
    assert "bilinear" in content
    assert "bicubic" in content
