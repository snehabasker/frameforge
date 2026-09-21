"""Render LR/HR training sequences with Combined/Depth/Vector passes.

Run headless (bpy is only importable inside Blender's own Python):

    blender -b <scene.blend> -P blender/render_sequences.py -- \
        --config configs/render.yaml --scene-name scene_a_courtyard \
        --out data/rendered

This script cannot be executed or import-tested outside Blender -- there is
no Blender install in the environment this codebase was built in (see
docs/DECISIONS.md D0). It has been written to the documented Blender Python
API and syntax-checked (`python -m py_compile`), but has NOT been run
end to end against a real .blend file. Treat the first real run as a
verification step, not a rubber stamp: watch for API drift across Blender
versions (this targets the 4.x bpy API) and confirm the EXR output channel
names match what `frameforge/data/exr_io.py` expects (configs/data.yaml's
exr_channel_map) -- adjust one or the other if they don't.

Design notes (CLAUDE.md 3.1):
  - LR and HR are each rendered NATIVELY at their own resolution in separate
    passes over the same animated camera, not one downsampled from the
    other -- this reproduces real aliasing at LR instead of removing it.
  - Motion blur is OFF (`render.use_motion_blur = False`): required for the
    Vector pass to mean anything (motion blur pre-integrates exactly the
    signal the Vector pass is meant to expose per-pixel).
  - Output layout matches what `frameforge/data/dataset.py` and the
    committed test fixtures already use:
        <out>/<scene_name>/<sequence_name>/lr/frame_XXXX.exr
        <out>/<scene_name>/<sequence_name>/hr/frame_XXXX.exr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

try:
    import bpy
except ImportError:  # pragma: no cover - only importable inside Blender
    bpy = None


def parse_args():
    # Blender passes its own argv up to a literal "--"; only what's after
    # that belongs to this script.
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to configs/render.yaml")
    parser.add_argument("--scene-name", required=True, help="entry under `scenes` in the config")
    parser.add_argument("--out", required=True, help="output root directory")
    return parser.parse_args(argv)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def find_scene_config(cfg: dict, scene_name: str) -> dict:
    for entry in cfg["scenes"]:
        if entry["name"] == scene_name:
            return entry
    raise KeyError(
        f"scene {scene_name!r} not found in config; add it under `scenes` in configs/render.yaml"
    )


def configure_render_settings(cfg: dict):
    """Cycles engine, sampling, denoising, motion blur OFF, EXR output."""
    scene = bpy.context.scene
    r = cfg["render"]

    scene.render.engine = r["engine"]
    scene.cycles.samples = r["samples"]
    scene.cycles.use_denoising = r["use_denoising"]
    scene.render.use_motion_blur = r["use_motion_blur"]
    if scene.render.use_motion_blur:
        raise ValueError(
            "use_motion_blur must be False -- required for a meaningful "
            "Vector pass (see module docstring)"
        )
    scene.view_settings.view_transform = r.get("color_management", "Standard")

    scene.render.image_settings.file_format = cfg["output"]["format"]
    scene.render.image_settings.color_depth = cfg["output"]["color_depth"]
    scene.render.image_settings.exr_codec = cfg["output"]["exr_codec"]


def configure_passes(cfg: dict):
    view_layer = bpy.context.view_layer
    passes = cfg["passes"]
    view_layer.use_pass_combined = passes.get("combined", True)
    view_layer.use_pass_z = passes.get("depth", True)
    view_layer.use_pass_vector = passes.get("vector", True)


def apply_camera_flythrough(cfg: dict, flythrough_name: str):
    """Keyframe the active camera's location/rotation per configs/render.yaml."""
    flythrough = cfg["camera_flythroughs"][flythrough_name]
    camera = bpy.context.scene.camera
    if camera is None:
        raise RuntimeError("scene has no active camera; set one in the .blend file")

    for kf in flythrough["keyframes"]:
        bpy.context.scene.frame_set(kf["frame"])
        camera.location = kf["location"]
        camera.rotation_euler = kf["rotation_euler"]
        camera.keyframe_insert(data_path="location", frame=kf["frame"])
        camera.keyframe_insert(data_path="rotation_euler", frame=kf["frame"])


def render_sequence_at_resolution(
    out_dir: Path, width: int, height: int, frame_start: int, frame_end: int
):
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.frame_start = frame_start
    scene.frame_end = frame_end

    out_dir.mkdir(parents=True, exist_ok=True)
    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        scene.render.filepath = str(out_dir / f"frame_{frame - frame_start:04d}.exr")
        bpy.ops.render.render(write_still=True)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    scene_cfg = find_scene_config(cfg, args.scene_name)

    configure_render_settings(cfg)
    configure_passes(cfg)
    apply_camera_flythrough(cfg, scene_cfg["camera_flythrough"])

    frame_count = scene_cfg["frame_count"]
    out_root = Path(args.out) / scene_cfg["name"] / scene_cfg["name"]

    lr_w, lr_h = cfg["resolutions"]["lr"]
    hr_w, hr_h = cfg["resolutions"]["hr"]

    print(f"[render_sequences] rendering LR {lr_w}x{lr_h}, {frame_count} frames -> "
          f"{out_root / 'lr'}")
    render_sequence_at_resolution(out_root / "lr", lr_w, lr_h, 1, frame_count)

    print(f"[render_sequences] rendering HR {hr_w}x{hr_h}, {frame_count} frames -> "
          f"{out_root / 'hr'}")
    render_sequence_at_resolution(out_root / "hr", hr_w, hr_h, 1, frame_count)

    print("[render_sequences] done")


if __name__ == "__main__":
    if bpy is None:
        raise RuntimeError(
            "this script must be run inside Blender: "
            "blender -b <scene.blend> -P blender/render_sequences.py -- "
            "--config ... --scene-name ... --out ..."
        )
    main()
