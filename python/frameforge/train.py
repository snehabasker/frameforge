"""Training loop: recurrent unroll, distillation losses, checkpoint/resume
(CLAUDE.md section 5.2-5.3).

`run_unroll` is the one place that implements "feed frames through
FrameForgeNet one at a time, in order, passing each output back in as the
next frame's prev_hr_output" -- used by the M2 overfit sanity test, the
full training loop below, and evaluate.py, so there is exactly one
recurrence implementation.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from frameforge.data.dataset import FrameSequenceDataset, SequenceBatch
from frameforge.losses import DistillationLosses, LossWeights
from frameforge.models.student import FrameForgeNet
from frameforge.models.teacher import SwinIRTeacher


def run_unroll(
    model: FrameForgeNet, sample: SequenceBatch, return_features: bool = False
):
    """Run `model` recurrently over one sequence's frames (batch size 1).

    Args:
        model: a FrameForgeNet (any variant).
        sample: a SequenceBatch with (T, C, H, W) tensors (as returned by
            FrameSequenceDataset).
        return_features: if True, also return the stacked per-frame
            mid-body features (model.last_feature at each step) for feature
            distillation. Without this, only the last frame's feature would
            survive the loop (model.last_feature gets overwritten every
            step), silently limiting feature KD to the final frame only --
            this flag exists specifically so callers doing feature
            distillation don't hit that trap.

    Returns:
        (T, 3, H*scale, W*scale) tensor of outputs O_0..O_{T-1}, WITH
        gradients able to flow through the whole chain (no detaching
        between steps) -- CLAUDE.md 5.2: "backprop through the unroll".
        If `return_features`, returns (outputs, features) instead, with
        features shaped (T, channels, H, W).
    """
    t_total = sample.lr_rgb.shape[0]
    prev_hr_output = None
    outputs = []
    features = []
    for t in range(t_total):
        lr_rgb_t = sample.lr_rgb[t:t + 1]
        kwargs = {}
        if model.use_depth:
            kwargs["lr_depth"] = sample.lr_depth[t:t + 1]
        if model.use_temporal:
            kwargs["lr_mv"] = sample.lr_mv[t:t + 1]
            kwargs["prev_hr_output"] = prev_hr_output
        out = model(lr_rgb_t, **kwargs)
        outputs.append(out)
        if return_features:
            features.append(model.last_feature)
        prev_hr_output = out
    stacked_out = torch.cat(outputs, dim=0)
    if return_features:
        return stacked_out, torch.cat(features, dim=0)
    return stacked_out


@dataclass
class TrainConfig:
    variant: str = "temporal"
    channels: int = 32
    num_blocks: int = 6
    lr: float = 2e-4
    total_steps: int = 1000
    checkpoint_every: int = 200
    loss_weights: LossWeights = None  # type: ignore[assignment]
    use_amp: bool = True
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        weights = LossWeights(**raw.pop("loss_weights", {}))
        return cls(loss_weights=weights, **raw)


def save_checkpoint(path: Path, model, optimizer, step: int, scaler=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "variant": model.variant,
            "channels": model.channels,
            "num_blocks": model.num_blocks,
        },
        path,
    )


def load_checkpoint(path: Path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=True)


def train(
    train_cfg: TrainConfig,
    data_root: Path,
    data_cfg: dict,
    scenes: list[str] | None,
    teacher_weights: Path | None,
    checkpoint_dir: Path,
    log_csv: Path,
    resume_from: Path | None = None,
    teacher_cache_dir: Path | None = None,
) -> FrameForgeNet:
    torch.manual_seed(train_cfg.seed)

    model = FrameForgeNet(
        variant=train_cfg.variant, channels=train_cfg.channels, num_blocks=train_cfg.num_blocks
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg.total_steps
    )
    # AMP/GradScaler only does anything on CUDA (Kaggle T4/P100, per
    # CLAUDE.md 1's compute budget); on CPU (this dev machine) it's
    # correctly a no-op, silenced explicitly here rather than left to
    # GradScaler's own runtime warning.
    amp_enabled = train_cfg.use_amp and torch.cuda.is_available()
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    start_step = 0
    if resume_from is not None and Path(resume_from).exists():
        ckpt = load_checkpoint(resume_from)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scaler_state"):
            scaler.load_state_dict(ckpt["scaler_state"])
        start_step = ckpt["step"] + 1
        print(f"resumed from {resume_from} at step {start_step}")

    needs_teacher = train_cfg.loss_weights.output_kd > 0 or train_cfg.loss_weights.feature_kd > 0
    teacher = None
    used_live_teacher_fallback = False
    if needs_teacher:
        if teacher_cache_dir is None and teacher_weights is None:
            raise ValueError(
                "loss_weights need a teacher (output_kd or feature_kd > 0) "
                "but neither teacher_cache_dir nor teacher_weights was given"
            )
        if teacher_cache_dir is not None:
            # Preferred path (CLAUDE.md section 4: "teacher inference never
            # runs inside the training loop") -- see
            # frameforge.data.teacher_cache.build_teacher_cache, run ahead
            # of time via `python -m frameforge.data.teacher_cache`. The
            # cache holds FULL, uncropped sequences; when the training
            # config also crops/windows (the spec's recommended setup,
            # CLAUDE.md 5.2), cached shapes won't line up with a given
            # step's crop, and this falls back to live inference for that
            # step only -- hence still wanting teacher_weights available
            # below as a fallback, not as the primary path.
            pass
        if teacher_weights is not None:
            teacher = SwinIRTeacher(teacher_weights)
        elif data_cfg["sequences"].get("crop_size") or data_cfg["sequences"].get("unroll_length"):
            raise ValueError(
                "teacher_cache_dir given without teacher_weights, but "
                "configs/data.yaml crops/windows sequences -- cached "
                "(uncropped) teacher tensors won't match a cropped batch's "
                "shape and there is no live teacher to fall back to. Either "
                "pass --teacher-weights too, or set crop_size/unroll_length "
                "to null to train on full, uncropped frames."
            )

    losses = DistillationLosses(train_cfg.loss_weights)

    dataset = FrameSequenceDataset(
        data_root, data_cfg["exr_channel_map"], data_cfg["tonemap"],
        scenes=scenes,
        unroll_length=data_cfg["sequences"]["unroll_length"],
        crop_size=data_cfg["sequences"]["crop_size"],
        seed=train_cfg.seed,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])

    log_csv.parent.mkdir(parents=True, exist_ok=True)
    log_new = not log_csv.exists()
    log_file = open(log_csv, "a", newline="")
    log_writer = csv.writer(log_file)
    if log_new:
        log_writer.writerow(["step", "loss_total", "loss_recon", "loss_output_kd",
                              "loss_feature_kd", "loss_temporal", "elapsed_s"])

    model.train()
    step = start_step
    data_iter = iter(loader)
    t_start = time.time()
    while step < train_cfg.total_steps:
        try:
            sample = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            sample = next(data_iter)

        optimizer.zero_grad()
        need_features = train_cfg.loss_weights.feature_kd > 0
        if need_features:
            out, student_feat = run_unroll(model, sample, return_features=True)
        else:
            out = run_unroll(model, sample)
            student_feat = None
        teacher_out = None
        teacher_feat = None
        if teacher_cache_dir is not None:
            from frameforge.data.teacher_cache import load_teacher_cache

            cached_out, cached_feat = load_teacher_cache(
                teacher_cache_dir, sample.scene, sample.sequence
            )
            # The cache holds the FULL, uncropped/un-windowed sequence;
            # FrameSequenceDataset may have windowed/cropped `sample` itself
            # via its own RNG, so slicing here would drift out of sync. To
            # keep this correct without duplicating the dataset's windowing
            # logic, cached teacher tensors are only used when the dataset
            # was constructed with unroll_length=None and crop_size=None
            # (i.e. full frames) -- otherwise fall back to live inference
            # for this step, still correct, just not cache-accelerated.
            shapes_match = (
                cached_out.shape[-2:] == sample.hr_rgb.shape[-2:]
                and cached_out.shape[0] == sample.hr_rgb.shape[0]
            )
            if shapes_match:
                teacher_out, teacher_feat = cached_out, cached_feat
            elif teacher is not None:
                if not used_live_teacher_fallback:
                    print(
                        "cached teacher shape doesn't match this (cropped/"
                        "windowed) batch; falling back to live teacher "
                        "inference for such steps (see train()'s docstring "
                        "note on teacher_cache_dir)."
                    )
                    used_live_teacher_fallback = True
                with torch.no_grad():
                    teacher_out, teacher_feat = teacher(sample.lr_rgb)
            else:
                raise RuntimeError(
                    "cached teacher shape mismatch and no live teacher "
                    "fallback available -- this should have been caught "
                    "before training started; see the crop_size/"
                    "unroll_length check above train()'s main loop"
                )
        elif teacher is not None:
            with torch.no_grad():
                teacher_out, teacher_feat = teacher(sample.lr_rgb)
        loss_dict = losses(
            student_out=out, hr_target=sample.hr_rgb,
            teacher_out=teacher_out, teacher_feat=teacher_feat, student_feat=student_feat,
            lr_mv=sample.lr_mv,
        )

        scaler.scale(loss_dict["total"]).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        log_writer.writerow([
            step, loss_dict["total"].item(), loss_dict["recon"].item(),
            loss_dict.get("output_kd", torch.tensor(0.0)).item(),
            loss_dict.get("feature_kd", torch.tensor(0.0)).item(),
            loss_dict.get("temporal", torch.tensor(0.0)).item(),
            time.time() - t_start,
        ])
        log_file.flush()

        if step % train_cfg.checkpoint_every == 0 or step == train_cfg.total_steps - 1:
            save_checkpoint(checkpoint_dir / f"step_{step:06d}.pt", model, optimizer, step, scaler)
            save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, step, scaler)

        step += 1

    log_file.close()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--teacher-weights", default=None)
    parser.add_argument(
        "--teacher-cache-dir", default=None,
        help="precomputed teacher cache from frameforge.data.teacher_cache (preferred; "
             "see CLAUDE.md section 4)",
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--log-csv", default="runs/train_log.csv")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    train_cfg = TrainConfig.from_yaml(args.train_config)
    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)

    train(
        train_cfg=train_cfg,
        data_root=Path(args.data_root),
        data_cfg=data_cfg,
        scenes=args.scenes,
        teacher_weights=Path(args.teacher_weights) if args.teacher_weights else None,
        checkpoint_dir=Path(args.checkpoint_dir),
        log_csv=Path(args.log_csv),
        resume_from=Path(args.resume) if args.resume else None,
        teacher_cache_dir=Path(args.teacher_cache_dir) if args.teacher_cache_dir else None,
    )


if __name__ == "__main__":
    main()
