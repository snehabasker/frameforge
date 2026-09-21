"""scripts/gen_results_table.py tests."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gen_results_table import aggregate_by_model, format_table  # noqa: E402


def test_aggregate_averages_across_sequences_same_model():
    rows = [
        {"model": "bilinear", "psnr_rgb": "10.0", "ssim_rgb": "0.5"},
        {"model": "bilinear", "psnr_rgb": "20.0", "ssim_rgb": "0.7"},
    ]
    agg = aggregate_by_model(rows)
    assert agg["bilinear"]["psnr_rgb"] == 15.0
    assert abs(agg["bilinear"]["ssim_rgb"] - 0.6) < 1e-9


def test_aggregate_missing_metric_is_none():
    rows = [{"model": "bilinear", "psnr_rgb": "10.0"}]
    agg = aggregate_by_model(rows)
    assert agg["bilinear"]["lpips"] is None


def test_params_not_averaged_takes_first():
    rows = [
        {"model": "student", "params": "1000"},
        {"model": "student", "params": "1000"},
    ]
    agg = aggregate_by_model(rows)
    assert agg["student"]["params"] == 1000.0


def test_format_table_missing_metric_renders_dashes():
    agg = {"bilinear": {"psnr_rgb": 12.345, "params": None}}
    # fill in all expected keys as None except psnr_rgb
    from gen_results_table import METRIC_COLUMNS

    full = {key: None for key, _, _ in METRIC_COLUMNS}
    full["psnr_rgb"] = 12.345
    agg = {"bilinear": full}
    table = format_table(agg)
    assert "12.35 dB" in table
    assert "--" in table


def test_format_table_preferred_ordering():
    full = {key: 1.0 for key, _, _ in __import__("gen_results_table").METRIC_COLUMNS}
    agg = {"zzz_custom": dict(full), "bicubic": dict(full), "bilinear": dict(full)}
    table = format_table(agg)
    lines = [line for line in table.splitlines() if line.startswith("| bi") or "zzz" in line]
    names_in_order = [line.split("|")[1].strip() for line in lines]
    assert names_in_order.index("bilinear") < names_in_order.index("bicubic")
    assert names_in_order.index("bicubic") < names_in_order.index("zzz_custom")
