"""M3 acceptance smoke test: "the full [experiment] matrix has run."

Runs a handful of real training steps for each of the four A/B/C/D configs
against the committed synthetic fixture (not the real rendered dataset,
which doesn't exist in this environment -- see docs/DECISIONS.md D0/D5),
using the precomputed teacher cache end to end, and checks each run
produces a finite, non-crashing loss and a checkpoint. This proves the
training loop, loss wiring, teacher caching, and checkpointing all work
together for every point in the experiment matrix; it is NOT a claim about
model quality on real data.
"""

from __future__ import annotations

import copy
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

CONFIG_NAMES = [
    "A_scratch", "B_output_kd", "C_output_feature_kd", "D_no_temporal",
]


@pytest.fixture(scope="module")
def teacher_cache_dir(tmp_path_factory):
    from frameforge.data.teacher_cache import build_teacher_cache
    from frameforge.models.teacher import SwinIRTeacher

    with open(REPO_ROOT / "configs" / "data.yaml") as f:
        data_cfg = yaml.safe_load(f)
    out_dir = tmp_path_factory.mktemp("teacher_cache")
    teacher = SwinIRTeacher(WEIGHTS_PATH)
    build_teacher_cache(
        FIXTURES / "sequences", data_cfg["exr_channel_map"], data_cfg["tonemap"], teacher, out_dir
    )
    return out_dir


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_experiment_matrix_run_smoke(config_name, teacher_cache_dir, tmp_path):
    from frameforge.train import TrainConfig, train

    train_cfg = TrainConfig.from_yaml(REPO_ROOT / "configs" / f"train_{config_name}.yaml")
    train_cfg.total_steps = 3  # smoke test, not a real run
    train_cfg.checkpoint_every = 3

    with open(REPO_ROOT / "configs" / "data.yaml") as f:
        data_cfg = yaml.safe_load(f)
    data_cfg = copy.deepcopy(data_cfg)
    # Uncropped/un-windowed so the (uncropped) teacher cache lines up exactly
    # -- see train.py's teacher_cache_dir docstring note.
    data_cfg["sequences"]["crop_size"] = None
    data_cfg["sequences"]["unroll_length"] = None

    model = train(
        train_cfg=train_cfg,
        data_root=FIXTURES / "sequences",
        data_cfg=data_cfg,
        scenes=None,
        teacher_weights=None,
        checkpoint_dir=tmp_path / "checkpoints",
        log_csv=tmp_path / "log.csv",
        teacher_cache_dir=teacher_cache_dir,
    )

    for p in model.parameters():
        assert torch.isfinite(p).all(), "non-finite parameter after training"
    assert (tmp_path / "checkpoints" / "latest.pt").exists()
    assert (tmp_path / "log.csv").exists()
