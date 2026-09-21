"""Teacher cache build + load round trip (CLAUDE.md section 4)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
WEIGHTS_PATH = REPO_ROOT / ".weights_cache" / "swinir_classicalSR_x2.pth"

pytestmark = pytest.mark.skipif(
    not WEIGHTS_PATH.exists(),
    reason="SwinIR weights not cached; run scripts/download_pretrained.py",
)


def test_build_and_load_teacher_cache(tmp_path):
    from frameforge.data.teacher_cache import build_teacher_cache, load_teacher_cache
    from frameforge.models.teacher import FEATURE_CHANNELS, SwinIRTeacher

    with open(REPO_ROOT / "configs" / "data.yaml") as f:
        cfg = yaml.safe_load(f)

    teacher = SwinIRTeacher(WEIGHTS_PATH)
    written = build_teacher_cache(
        FIXTURES / "sequences", cfg["exr_channel_map"], cfg["tonemap"], teacher, tmp_path
    )
    assert len(written) == 1

    teacher_out, teacher_feat = load_teacher_cache(tmp_path, "synthetic_pan_00", "synthetic_pan_00")
    assert teacher_out.shape == (3, 3, 72, 128)
    assert teacher_feat.shape[0] == 3 and teacher_feat.shape[1] == FEATURE_CHANNELS
    assert teacher_out.dtype == torch.float32  # loader upcasts from the fp16 cache
