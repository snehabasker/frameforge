"""SwinIR teacher wrapper tests.

Skipped when the pretrained weights aren't cached locally (run
`python scripts/download_pretrained.py` first) -- CI intentionally does not
fetch this 64MB checkpoint (CLAUDE.md section 8: "CI must not require a GPU
or large data"), so this test only runs in a dev environment that has it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = REPO_ROOT / ".weights_cache" / "swinir_classicalSR_x2.pth"

pytestmark = pytest.mark.skipif(
    not WEIGHTS_PATH.exists(),
    reason="SwinIR weights not cached; run scripts/download_pretrained.py",
)


def test_teacher_loads_and_is_frozen():
    from frameforge.models.teacher import SwinIRTeacher

    teacher = SwinIRTeacher(WEIGHTS_PATH)
    assert not any(p.requires_grad for p in teacher.parameters())
    assert not teacher.net.training  # eval mode


def test_teacher_forward_shapes_and_range():
    from frameforge.models.teacher import FEATURE_CHANNELS, SwinIRTeacher

    teacher = SwinIRTeacher(WEIGHTS_PATH)
    x = torch.rand(1, 3, 36, 64)
    sr, feat = teacher(x)

    assert sr.shape == (1, 3, 72, 128)
    assert sr.min() >= 0.0 and sr.max() <= 1.0
    assert feat.shape[0] == 1 and feat.shape[1] == FEATURE_CHANNELS


def test_teacher_is_deterministic():
    from frameforge.models.teacher import SwinIRTeacher

    teacher = SwinIRTeacher(WEIGHTS_PATH)
    torch.manual_seed(0)
    x = torch.rand(1, 3, 24, 24)
    sr1, _ = teacher(x)
    sr2, _ = teacher(x)
    assert torch.equal(sr1, sr2)
