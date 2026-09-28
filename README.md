# FrameForge

A small, distilled, temporally stable neural upscaler for real-time rendered frames (960x540 -> 1920x1080, x2), trained in PyTorch and shipped as a multithreaded C++ inference engine.

**Status: work in progress.** This README is a stub populated at each milestone; see `docs/DECISIONS.md` for the running log of design decisions and `docs/RESULTS.md` for auto-generated results once training has run.

## Milestones

| # | Milestone | Status |
|---|---|---|
| M0 | Skeleton | done |
| M1 | Data | code + tests done; real 40-sequence Blender render pending (needs Blender) |
| M2 | Teacher + student | done (real SwinIR teacher, param-budget-enforced student, overfit sanity check all verified) |
| M3 | Distillation + eval | code + full pipeline verified end to end (training matrix, evaluate.py, FSR1/bilinear/bicubic/SwinIR baselines, results table, failure crops) on the synthetic fixture; real 40-sequence matrix run pending real data |
| M4 | Export + engine core | done (ONNX export + parity verified; C++ engine built with real ONNX Runtime, tinyexr, runs a sequence end to end, output verified byte-identical to PyTorch) |
| M5 | Concurrency + perf | done (threaded reader/inference/writer pipeline, both queues implemented + benchmarked, tiled inference, 31/31 GoogleTest passing locally; ASan/TSan/flamegraph need CI's Linux runners, not verifiable on this Windows dev machine) |
| M6 | Polish | not started |

Full spec: see `CLAUDE.md`.

## Reproduction environment note

This repository was developed on a Windows machine without Blender or a Kaggle GPU session attached to the coding environment itself. Every stage is scripted and reproducible, but the two compute-heavy stages -- rendering the training dataset in Blender, and the full distillation training matrix on a GPU -- must be run by a human on a machine that has those resources. See `docs/DECISIONS.md` for the exact, honest scope reduction this causes and the commands to run each stage.
