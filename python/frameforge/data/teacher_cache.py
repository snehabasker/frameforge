"""Precompute and cache frozen-teacher outputs/features (CLAUDE.md section 4:
"teacher inference never runs inside the training loop").

One .pt file per sequence, keyed the same way frameforge.data.cache does,
so train.py can look a sequence up by (scene, sequence) and get back
fp16 teacher_out/teacher_feat tensors instead of calling the teacher live.
Loads full (uncropped, un-windowed) sequences so the cache is reusable
across different crop/unroll training configs -- train.py crops/windows the
cached tensors the same way it would a freshly-loaded one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from frameforge.data.dataset import discover_sequences
from frameforge.data.preprocess import load_lr_sequence
from frameforge.models.teacher import SwinIRTeacher


def build_teacher_cache(
    root: str | Path,
    channel_map: dict,
    tonemap_cfg: dict,
    teacher: SwinIRTeacher,
    out_dir: str | Path,
    scenes: list[str] | None = None,
) -> list[Path]:
    """Run the frozen teacher once over every sequence under `root` and
    cache (teacher_out, teacher_feat) to disk in fp16 (CLAUDE.md section 4:
    "Store features in fp16 to fit disk limits").
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for scene, sequence, seq_dir in discover_sequences(Path(root), scenes=scenes):
        lr = load_lr_sequence(seq_dir / "lr", channel_map, tonemap_cfg)
        teacher_out, teacher_feat = teacher(lr["rgb"])
        payload = {
            "teacher_out": teacher_out.half(),
            "teacher_feat": teacher_feat.half(),
            "scene": scene,
            "sequence": sequence,
        }
        out_path = out_dir / f"{scene}__{sequence}.pt"
        torch.save(payload, out_path)
        written.append(out_path)
    return written


def load_teacher_cache(cache_dir: str | Path, scene: str, sequence: str):
    path = Path(cache_dir) / f"{scene}__{sequence}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["teacher_out"].float(), payload["teacher_feat"].float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="dir of scene/sequence/{lr,hr} EXRs")
    parser.add_argument("--out", required=True, help="output dir for teacher cache .pt files")
    parser.add_argument("--teacher-weights", required=True)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    teacher = SwinIRTeacher(args.teacher_weights)
    written = build_teacher_cache(
        args.root, cfg["exr_channel_map"], cfg["tonemap"], teacher, args.out
    )
    print(f"wrote {len(written)} teacher cache file(s) to {args.out}")


if __name__ == "__main__":
    main()
