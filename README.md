# FrameForge

A small, distilled, temporally stable neural upscaler for real-time rendered frames (960x540 -> 1920x1080, x2), trained in PyTorch and shipped as a multithreaded C++ inference engine.

**Status: work in progress.** This README is a stub populated at each milestone; see `docs/DECISIONS.md` for the running log of design decisions and `docs/RESULTS.md` for auto-generated results once training has run.

## Milestones

| # | Milestone | Status |
|---|---|---|
| M0 | Skeleton | done |
| M1 | Data | code + tests done; real 40-sequence Blender render pending (needs Blender) |
| M2 | Teacher + student | not started |
| M3 | Distillation + eval | not started |
| M4 | Export + engine core | not started |
| M5 | Concurrency + perf | not started |
| M6 | Polish | not started |

Full spec: see `CLAUDE.md`.

## Reproduction environment note

This repository was developed on a Windows machine without Blender or a Kaggle GPU session attached to the coding environment itself. Every stage is scripted and reproducible, but the two compute-heavy stages -- rendering the training dataset in Blender, and the full distillation training matrix on a GPU -- must be run by a human on a machine that has those resources. See `docs/DECISIONS.md` for the exact, honest scope reduction this causes and the commands to run each stage.
