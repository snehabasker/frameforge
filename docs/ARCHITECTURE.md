# Architecture

## Overview

FrameForge has two halves that meet at one file: an ONNX graph.

```
Python (train.py, export_onnx.py)              C++ (engine/)
─────────────────────────────────              ─────────────
FrameForgeNet (PyTorch)                         OrtModelRunner (ONNX Runtime C++)
  warp() inside forward()          --export-->    the SAME warp, as ONNX GridSample
  teacher distillation losses                    sequence_runner.cpp / pipeline.cpp
  recurrent unroll (run_unroll)                    recurrent inference loop
                                                   tile.cpp (tiled inference)
                                                   queue.hpp (BlockingQueue, SpscRingBuffer)
```

The model's forward pass -- including the previous-frame warp -- is traced into the ONNX graph once (`export_onnx.py`). From that point on, the C++ engine only feeds tensors in and reads tensors out (`frameforge::OrtModelRunner::Run`); it never reimplements the warp, the residual blocks, or any model math. `engine/tests/engine_e2e_test.cpp` proves this boundary holds: it runs the SAME exported model through both PyTorch (`run_unroll`) and the C++ engine's full pipeline (EXR read -> tonemap -> depth-normalize -> recurrent ONNX inference) on a real sequence and asserts the outputs match within tolerance.

## The recurrent dependency, and where concurrency actually comes from

`O_t` depends on `O_{t-1}` (the model warps the previous output using this frame's motion vectors). That means **frames within one sequence must be inferred in strict order** -- there is no way to parallelize inference across a single sequence's timeline without breaking correctness. This shapes the whole concurrency design:

- **Parallelism across sequences**: `RunSequencesConcurrently` (`pipeline.cpp`) runs `--threads` independent sequences at once, each with its own `OrtModelRunner` (the models are small enough -- under 300K params -- that duplicating one session per thread is cheaper than reasoning about ONNX Runtime's concurrent-session semantics). This is where `--threads > 1` actually buys throughput.
- **Parallelism across pipeline stages, within one sequence**: `RunSequencePipelined` splits each sequence's processing into reader / inference / writer stages on three threads, connected by bounded queues. Inference for frame *t+1* can proceed while the writer PNG-encodes frame *t*'s output, and reading can proceed ahead of inference -- **with one real limit**: per-sequence depth normalization (CLAUDE.md 3.3, mirrored exactly from `frameforge.colorspace.normalize_depth`) needs every frame's depth range before any frame can be normalized. So the reader stage cannot start handing frames to the inference stage until it has decoded the whole sequence; reading and inference do not overlap with each other, only inference and writing do. This was discovered while building the pipeline, not designed in from the start -- see `docs/DECISIONS.md` D9 for the full reasoning. `engine/tests/pipeline_test.cpp` confirms the pipelined path is still byte-identical to the sequential one; overlapping doesn't change what gets computed, only when.

## Queues

Two implementations exist (`engine/include/frameforge/queue.hpp`):

- `BlockingQueue<T>`: mutex + condition variable, general-purpose, supports multiple producers/consumers and a clean `Close()` shutdown.
- `SpscRingBuffer<T>`: lock-free, single-producer/single-consumer, fixed capacity (rounded to a power of two), `std::atomic` with explicit acquire/release memory orders (reasoning is commented inline in the header: the producer publishes a slot's data via a release store to `head_`, the consumer's acquire load on `head_` makes that data visible before it reads the slot).

`engine/bench/queue_bench.cpp` benchmarks both under the same single-producer/single-consumer workload; `SpscRingBuffer` wins by ~1.2x-5.6x depending on item count (see `docs/RESULTS.md`), and is the pipeline's default. `BlockingQueue` stays available (and tested) for cases needing multiple producers/consumers or a `Close()`-based shutdown signal, which `SpscRingBuffer` doesn't support natively.

## Tiled inference

`RunTiled` (`tile.cpp`) splits an LR frame into overlapping tiles, runs each through the model independently, and blends the overlaps back together with linear feathering (full weight in a tile's interior, ramping down over the overlap region at an edge that borders another tile; edges that touch the true frame boundary stay at full weight, since there's nothing to blend with there). `prev_hr_output` is sliced at the corresponding HR region per tile, so temporal models tile correctly too. `engine/tests/tile_test.cpp` checks tiled output stays within tolerance of the untiled path.

## Data flow through one frame

1. **Reader**: `LoadFramePasses` (`exr_io.cpp`, via vendored tinyexr) reads the Combined/Depth/Vector passes from a multilayer EXR, matching the channel-naming convention `python/frameforge/data/exr_io.py` uses.
2. **Preprocess** (`preprocess.cpp`, deliberately mirrored 1:1 against `python/frameforge/colorspace.py` -- see `engine/tests/tonemap_test.cpp`'s cross-checked values): Reinhard tonemap + sRGB encode on Combined; per-sequence min-max normalize on Depth; zero the Vector pass on a sequence's first frame (no previous frame exists).
3. **Inference**: `OrtModelRunner::Run` feeds `(lr_rgb, lr_depth, lr_mv, prev_hr_output)` through the ONNX graph. The graph's own `GridSample` op does the previous-output warp -- the engine passes the RAW previous output, never pre-warping it itself (a real bug caught and fixed while building this: warping both in C++ and again inside the graph silently doubles the motion compensation).
4. **Writer**: `WritePng` (`png_io.cpp`, via vendored stb_image_write) clamps to `[0, 1]` and writes 8-bit PNG.

## What's not built yet

- CUDA execution provider support exists in `OrtModelRunner`'s constructor (`--ep cuda`) but is unexercised -- no CUDA-capable ONNX Runtime build or GPU driver setup was available while building this.
- ASan/UBSan/TSan are configured (`-DFRAMEFORGE_ENABLE_ASAN=ON` / `-DFRAMEFORGE_ENABLE_TSAN=ON`) but could not be verified locally -- this dev machine's MinGW-w64 toolchain ships the sanitizer headers but not their runtime libraries (docs/DECISIONS.md D8). CI's Ubuntu `cpp-asan-ubsan`/`cpp-tsan` jobs are the authoritative check, not yet confirmed green (no GitHub remote has been pushed to from this environment).
- Flamegraph generation (`scripts/flamegraph.sh`) needs Linux `perf`, unavailable on this Windows dev machine; the script is written and documented but not run here.
