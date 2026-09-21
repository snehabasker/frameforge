"""M2 acceptance (CLAUDE.md): "student trains end to end on a small subset
and overfits a single sequence (sanity check)".

Runs a real (tiny, fast) training loop -- recurrent unroll, backprop
through time, Adam -- against the committed synthetic fixture sequence, and
asserts the reconstruction loss drops sharply. This is the actual mechanism
train.py uses (see frameforge.train.run_unroll), exercised end to end here
against data small enough to run in a few seconds under pytest/CI.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from frameforge.data.dataset import FrameSequenceDataset
from frameforge.models.student import FrameForgeNet
from frameforge.train import run_unroll

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_student_overfits_single_synthetic_sequence():
    with open(REPO_ROOT / "configs" / "data.yaml") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(0)
    ds = FrameSequenceDataset(FIXTURES / "sequences", cfg["exr_channel_map"], cfg["tonemap"])
    sample = ds[0]

    model = FrameForgeNet(variant="temporal", channels=32, num_blocks=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    with torch.no_grad():
        initial_out = run_unroll(model, sample)
        initial_loss = F.l1_loss(initial_out, sample.hr_rgb).item()

    model.train()
    for _ in range(300):
        optimizer.zero_grad()
        out = run_unroll(model, sample)
        loss = F.l1_loss(out, sample.hr_rgb)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_out = run_unroll(model, sample)
        final_loss = F.l1_loss(final_out, sample.hr_rgb).item()

    assert final_loss < initial_loss * 0.2, (
        f"expected the model to overfit this single sequence: "
        f"initial_loss={initial_loss:.4f}, final_loss={final_loss:.4f}"
    )
    assert final_loss < 0.03, f"absolute overfit loss too high: {final_loss:.4f}"
