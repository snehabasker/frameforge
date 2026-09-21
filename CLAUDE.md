# FrameForge — Project Specification for Coding Agent

## 0. Your role

You are the implementation partner on **FrameForge**: a small, distilled, temporally stable neural upscaler for real-time rendered frames (960×540 → 1920×1080, ×2), trained in PyTorch and shipped as a multithreaded C++ inference engine.

The project exists to demonstrate, with measurable evidence, five things: model architecture design, knowledge distillation, ML for graphics (image upscaling), production C++ (CMake, CI, concurrency, profiling), and disciplined engineering. Quality of engineering matters as much as model quality.

Work milestone by milestone (Section 9). Do not start a milestone until the previous one meets its acceptance criteria.

---

## 1. Hard constraints

- **Zero budget.** No paid APIs, no paid compute, no paid datasets.
  - Training runs on Kaggle free GPUs (T4 16 GB or P100, ~30 GPU-h/week).
  - Keep every training run ≤ 3 h wall-clock and resumable from checkpoints.
- **Licensing.** Use only CC0 / CC-BY / MIT / Apache-2.0 / BSD assets and code.
  - Record every third-party asset and its license in `THIRD_PARTY.md`.
- **No fabricated numbers.** Every metric, table, chart and claim in the README must be produced by a committed script from committed configs.
  - If a result is bad, report it. Do not tune the evaluation to flatter the model.
- **Reproducibility.** Fixed seeds, YAML configs, pinned dependency versions, and a single command per stage.
- **Ask before** adding any dependency not listed in this spec.

---

## 2. Repository layout

```
frameforge/
├── README.md                  # pitch, results table, comparison video/GIF, how to run
├── THIRD_PARTY.md
├── pyproject.toml             # Python package + pinned deps
├── configs/                   # YAML: data, student variants, training, eval
├── blender/
│   └── render_sequences.py    # bpy script: renders LR/HR sequences + passes
├── python/frameforge/
│   ├── data/                  # EXR loading, sequence dataset, caching
│   ├── models/
│   │   ├── student.py
│   │   └── teacher.py         # SwinIR wrapper, frozen
│   ├── warp.py                # motion-vector warping (grid_sample)
│   ├── losses.py              # recon, distillation, temporal
│   ├── metrics.py             # PSNR, SSIM, LPIPS, temporal error, params, FLOPs
│   ├── train.py
│   ├── evaluate.py
│   └── export_onnx.py
├── baselines/
│   └── fsr1_easu/             # CPU reference port of FSR 1 EASU (+RCAS), C++
├── engine/                    # C++ inference engine
│   ├── CMakeLists.txt
│   ├── include/frameforge/
│   ├── src/
│   ├── tests/                 # GoogleTest
│   └── bench/
├── scripts/                   # results tables, comparison video, flamegraphs
├── docs/
│   ├── ARCHITECTURE.md
│   ├── RESULTS.md
│   └── DECISIONS.md           # short ADR-style log of design choices
└── .github/workflows/         # CI
```

---

## 3. Data (Milestone M1)

### 3.1 Rendering

Write `blender/render_sequences.py`, run headless (`blender -b scene.blend -P render_sequences.py -- --config ...`). It must:

- Render 3–4 CC0/CC-BY scenes (e.g. Poly Haven assets, Blender demo files with compatible licenses) with scripted camera flythroughs: pans, dolly moves, fast rotation, and occlusion events (objects passing in front of others).
- Render each sequence **twice, natively**, at 960×540 (LR) and 1920×1080 (HR). Do not generate LR by downsampling HR; native low-res rendering reproduces real aliasing.
- Use Cycles with **motion blur disabled** (required for the vector pass). Use a low sample count with denoising on for HR ground truth.
- Enable and save the following passes as multilayer OpenEXR (float32/half, no tonemapping in stored data):
  - Combined (RGB)
  - Depth (Z)
  - Vector (motion)
- Target volume: ~40 sequences × 30–60 frames total. Split train/val/test **by sequence and scene**, never by frame. Hold out one entire scene for testing.

### 3.2 Motion-vector convention (must verify, do not assume)

Blender's vector pass stores per-pixel motion to the previous and next frames in pixel units, with sign and axis conventions that must be confirmed empirically. Write a test that:

1. warps frame t−1 into frame t using the LR vector pass, and
2. asserts the warped image matches frame t (low error) in non-occluded regions for a known camera pan.

Document the confirmed convention in `docs/DECISIONS.md`.

### 3.3 Preprocessing

- Convert EXR to cached tensors (`.npz` or `.pt`), stored as linear HDR.
- Apply a fixed tonemap (e.g. Reinhard or ACES approximation, documented) followed by sRGB encoding, **before** feeding models and computing metrics. Use the same transform everywhere.
- Normalize depth per sequence to [0, 1] (document the method).
- Optional: pretrain on DIV2K (single-frame, synthetic ×2) if rendered data proves insufficient. Log the decision.

**M1 acceptance:** rendering script is reproducible from config; warp-consistency test passes; dataset loader returns `(lr_rgb, lr_depth, lr_mv, hr_rgb)` sequences; a sample grid image is saved to `docs/`.

---

## 4. Teacher (Milestone M2)

- Use official **SwinIR** pretrained ×2 weights (classical SR; Apache-2.0, cite the repo).
- The teacher is frozen and single-frame, operating on LR RGB only.
- Precompute and cache teacher outputs (and chosen intermediate features, Section 5.3) for the training set so teacher inference never runs inside the training loop. Store features in fp16 to fit disk limits; downsample or subset if needed and document it.
- Report teacher metrics on the test set. The teacher is an upper reference, not a baseline to beat.

---

## 5. Student model and distillation (Milestones M2–M3)

### 5.1 Architecture: `FrameForgeNet`

**Inputs at time t, all at LR resolution unless stated:**
- LR RGB, 3 ch
- LR depth, 1 ch
- Previous HR output O_{t−1}, warped to frame t using motion vectors upsampled to HR, then pixel-unshuffled (×2) back to LR resolution → 12 ch
- Optional disocclusion/validity mask derived from a depth or motion-vector consistency check, 1 ch

**Body:**
- 3×3 conv → N residual blocks (conv–ReLU–conv, no batch norm) at C channels → 3×3 conv to 12 ch → PixelShuffle(×2)
- Add a global skip: bilinear upsample of LR RGB.

**Configuration:**
- Default: C=32, N=6.
- Hard budget: **≤ 300K parameters**. Report params and GFLOPs at 540p input.
- First frame of a sequence: previous output = bilinear upsample of LR RGB.
- Put the warp (`grid_sample`) **inside the model's forward pass** so it exports to ONNX (GridSample, opset ≥ 16). The C++ engine then only feeds tensors.

**Required variants (config flags, same training budget):**
- `single_frame`: no temporal input (depth optional)
- `temporal`: full inputs
- `temporal_nodepth`: ablation

### 5.2 Training

- Train on recurrent unrolls: sequences of K=6 frames with random crops (e.g. 128×128 LR), backprop through the unroll.
- Use AdamW with cosine decay and mixed precision (AMP).
- Checkpoint every N steps. Every run must resume cleanly after a Kaggle session ends.
- Log to CSV/TensorBoard, not to paid trackers.

### 5.3 Losses and distillation variants

- **Reconstruction:** L1(O_t, HR_t)
- **Output distillation:** L1(O_t, Teacher_t), weight λ_out
- **Feature distillation:** L1/MSE between the student's mid-body feature map (projected by a 1×1 conv adapter to the teacher's channel count, spatially aligned) and a chosen SwinIR intermediate feature. The adapter is training-only and **must not** be exported.
- **Temporal consistency:** L1(O_t, warp(O_{t−1})) masked to valid (non-disoccluded) pixels, weight λ_temp. Its effect must be measured, not assumed.

**Distillation experiment matrix** (same architecture, same steps, same seed set; ≥ 2 seeds each if compute allows):

| Run | Recon | Output KD | Feature KD | Temporal |
|---|---|---|---|---|
| A — scratch | ✓ | | | ✓ |
| B — output KD | ✓ | ✓ | | ✓ |
| C — output + feature KD | ✓ | ✓ | ✓ | ✓ |
| D — C without temporal loss | ✓ | ✓ | ✓ | |

**M2 acceptance:** teacher cached; student trains end to end on a small subset and overfits a single sequence (sanity check); param budget enforced by a unit test.
**M3 acceptance:** the full matrix has run; results are auto-generated into `docs/RESULTS.md`.

---

## 6. Evaluation (Milestone M3)

`evaluate.py` runs every model and baseline on the held-out test scene and writes `results/*.csv`, plus a markdown table generated by `scripts/gen_results_table.py`.

**Metrics:**
- PSNR and SSIM on RGB and on Y channel (sRGB, after the fixed tonemap)
- LPIPS (AlexNet)
- **Temporal error:** mean |O_t − warp(O_{t−1})| over valid pixels, reported alongside the same quantity for ground truth (the gap is what matters)
- Parameters, GFLOPs, and ONNX Runtime latency (Section 7)

**Baselines:**
- Bilinear
- Bicubic
- **FSR 1 (EASU + RCAS)**
- SwinIR teacher (upper reference)
- All student variants

**Failure-case analysis:**
- Automatically extract crops with the worst error: disocclusions, thin geometry, fast motion, specular highlights.
- Save before/after crops to `docs/failures/` and write short honest notes in `RESULTS.md`.

### 6.1 FSR 1 baseline

- Implement a **CPU reference port** of FSR 1 EASU and RCAS in C++ under `baselines/fsr1_easu/`, faithfully following the official MIT-licensed `ffx_fsr1.h` from AMD's FidelityFX SDK. Keep the attribution and license notice.
- Expose it via a CLI (and pybind11 or a subprocess for evaluation).
- Validate it visually and document known deviations.
- **Framing rule:** this port is for **quality comparison only**. Never report its CPU speed as FSR 1 performance; the real FSR 1 runs as a GPU shader.

---

## 7. C++ inference engine (Milestones M4–M5)

### 7.1 Stack
- C++17 (C++20 allowed if justified in `DECISIONS.md`), CMake ≥ 3.20
- ONNX Runtime C++ API (prebuilt release fetched via CMake `FetchContent` or a documented download script), CPU EP by default, CUDA EP optional behind a flag
- tinyexr (EXR) and stb_image / stb_image_write (PNG) for I/O
- GoogleTest for tests, Google Benchmark for microbenchmarks

### 7.2 Export (`export_onnx.py`)
- Export the student with the warp inside the graph, dynamic H/W axes, and no training adapters.
- **Parity test:** PyTorch vs ONNX Runtime (Python) max abs error < 1e-4 on fixed inputs. This test must pass in CI.

### 7.3 Engine design

**CLI:**
```
frameforge --model m.onnx --input <seq_dir> --output <dir> \
           --threads N --ep cpu|cuda [--tile 256 --overlap 16] [--bench]
```

**Pipeline:** three stages connected by **bounded** queues.
- **Reader:** load EXR and preprocess
- **Inference:** ONNX Runtime session, recurrent state held here
- **Writer:** postprocess and write PNG/EXR

**Concurrency rules:**
- The recurrent dependency (O_{t−1} → O_t) means **frames within one sequence are inferred strictly in order**. Parallelism comes from overlapping pipeline stages and from processing independent sequences concurrently (one inference worker per sequence, `--threads` controls the pool).
- Document this reasoning in `ARCHITECTURE.md`.

**Queues:**
- Implement a mutex + condition-variable bounded blocking queue.
- Also implement a lock-free SPSC ring buffer (atomics with explicit memory orders, commented).
- Benchmark both and keep the better one as default, with the numbers in `RESULTS.md`.

**Other requirements:**
- **Tiled inference:** optional tiling with overlap and blending for inputs larger than memory allows, with a test that the tiled output matches the untiled output within tolerance.
- **Error handling:** no exceptions crossing thread boundaries silently. Clean shutdown on error or SIGINT, and no leaks under ASan.
- **Code style:** RAII throughout, no raw `new`/`delete`, `const`-correctness, `clang-format` config committed, `clang-tidy` clean on the chosen checks.

### 7.4 Tests (GoogleTest)
- Queue correctness under contention, including a stress test run under TSan
- Tensor layout conversion (HWC ↔ NCHW, tonemap round-trip)
- Engine output vs Python reference on a fixed sequence (tolerance-based)
- Tiled vs untiled equivalence
- FSR 1 port: deterministic output on a fixed image (golden file)

### 7.5 Benchmarks and profiling
- Report median and p95 ms/frame at 540p → 1080p for: 1 vs N threads, CPU EP vs CUDA EP (if available), and both queue implementations.
- Always record the hardware and software versions next to the numbers.
- Generate CPU flamegraphs with `perf` + FlameGraph scripts (`scripts/flamegraph.sh`) and commit the SVGs to `docs/`.
- If run on an AMD CPU, add an AMD uProf hotspot capture as well. Optional Tracy zones behind a compile flag.
- Write a short "where the time goes" analysis and one optimization made because of profiling, with before/after numbers.

---

## 8. CI (GitHub Actions)

**Jobs, on push and PR:**
1. **cpp-release:** configure, build, run tests (Ubuntu, GCC and Clang matrix)
2. **cpp-asan-ubsan:** build with `-fsanitize=address,undefined`, run tests
3. **cpp-tsan:** build with `-fsanitize=thread`, run the concurrency tests
4. **cpp-lint:** `clang-format --dry-run --Werror`, `clang-tidy`
5. **python:** `ruff`, `pytest` (param budget, warp-convention test on a tiny fixture, ONNX parity on a tiny exported model)

**CI rules:**
- CI must not require a GPU or large data. Keep tiny fixtures (a few small frames) in `tests/fixtures/`.
- Cache ONNX Runtime downloads and pip wheels.

---

## 9. Milestones

| # | Milestone | Acceptance criteria |
|---|---|---|
| M0 | Skeleton | Repo layout, pyproject, CMake builds an empty target, CI green, README stub |
| M1 | Data | Section 3 acceptance met |
| M2 | Teacher + student | Section 5 M2 acceptance met |
| M3 | Distillation + eval | Full experiment matrix, all baselines incl. FSR 1 port, auto-generated `RESULTS.md`, failure crops |
| M4 | Export + engine core | ONNX parity test passing; engine runs a sequence end to end; tests pass under ASan |
| M5 | Concurrency + perf | Pipeline, both queues benchmarked, TSan clean, flamegraphs + one profiling-driven optimization |
| M6 | Polish | README with results table, side-by-side comparison video/GIF (bicubic │ FSR 1 │ FrameForge │ GT, via ffmpeg script), ARCHITECTURE.md complete, one-command reproduction instructions |

---

## 10. README requirements (M6)

Order:
1. One-sentence pitch
2. Comparison GIF
3. Results table (quality + latency + params)
4. Key findings, 3–5 bullets, including **where the model loses**
5. Architecture diagram
6. How to reproduce
7. Engine design summary
8. Limitations and future work
9. Licenses and attribution

Every number in the README comes from `docs/RESULTS.md`, and every number in `RESULTS.md` comes from a script.

---

## 11. Working rules for you (the agent)

- Propose a short plan before each milestone. Wait for approval on anything that changes this spec.
- Small, focused commits with descriptive messages; one milestone per branch/PR.
- Write tests alongside code, not after. C++ utilities get tests first.
- When unsure about a convention (motion vectors, color space, ONNX op support), write a test that establishes it rather than assuming.
- Log non-trivial design decisions in `docs/DECISIONS.md` (context → decision → consequence, 3–6 lines each).
- Never hardcode local paths or credentials. Never commit data, checkpoints or large binaries (use `.gitignore`; publish weights separately).
- If a compute limit makes something infeasible, say so and propose the smallest honest reduction in scope.
