# Design decisions (ADR-style log)

Format: context -> decision -> consequence, 3-6 lines each. Newest at bottom.

## D0. Development environment constraints (honest scope reduction)

**Context:** The coding environment for this project is a Windows machine with no Blender install, no WSL, no GPU-enabled PyTorch by default (an RTX 5050 laptop GPU is present in `nvidia-smi` but the installed torch build is CPU-only), and no C/C++ compiler or CMake preinstalled. The spec (CLAUDE.md section 1) forbids fabricated numbers and requires every metric to come from a committed script run on committed configs.

**Decision:** Build the entire codebase to the full spec -- every script, model, loss, the C++ engine, tests, CI -- as real, working, tested code. Verify everything that can be verified locally: unit tests, the ONNX parity check, an overfit sanity check, and (once a local toolchain is installed, see D1) compiled-and-run C++ tests and benchmarks. For the two stages that need resources this machine does not have -- the 40-sequence Blender render (needs Blender + CC0 scene assets) and the full distillation training matrix at real scale (needs a Kaggle T4/P100 session per the spec's own compute budget) -- ship the exact one-command scripts to run them, and use tiny synthetic fixtures under `tests/fixtures/` to prove the pipeline logic is correct end to end. `docs/RESULTS.md` marks any number that depends on those two stages as **PENDING** with the exact command to produce it, rather than inventing a value.

**Consequence:** Every number that does appear in this repo's docs is real and reproducible from a committed script. The results table cannot be fully populated until a human runs the render and training stages on appropriate hardware. This is the smallest honest reduction in scope available, per CLAUDE.md section 11 ("If a compute limit makes something infeasible, say so and propose the smallest honest reduction in scope").

## D1. Local C++ toolchain: standalone MinGW-w64 GCC 16.2.0, not the conda-forge `m2w64-toolchain`

**Context:** `conda install -c conda-forge m2w64-toolchain` provides GCC 5.3.0, which predates standardized `<filesystem>`, lacks `if constexpr` support quality needed, and is effectively pre-C++17-complete. Compiling against it failed immediately (`fatal error: filesystem: No such file or directory`).

**Decision:** Downloaded a standalone WinLibs MinGW-w64 build (GCC 16.2.0, POSIX threads, UCRT runtime, x86_64) from the `brechtsanders/winlibs_mingw` GitHub releases (public, MIT-compatible redistribution of upstream GCC/MinGW-w64) and installed it at `C:\mingw64`, outside the repository. `cmake` and `ninja` still come from a dedicated `frameforge-cpp` conda environment.

**Consequence:** Full C++17 (and C++20) support locally, matching what GitHub Actions' Ubuntu runners provide via apt gcc/clang. This toolchain is a personal machine setup step, not a project dependency -- CI does not depend on it and uses its own Ubuntu-provided compilers per section 8.

## D2. Toolchain install path cannot contain spaces

**Context:** The repository lives under `C:\Users\Sneha Basker\Downloads\frameforge` -- a path containing a space. Installing the MinGW toolchain under that tree caused `ld.exe` to fail resolving its own internal default-manifest object path, splitting the path at the space (`cannot find C:\Users\Sneha: No such file or directory`). This is a known MinGW-w64/binutils issue with spaces in installation paths, not in project paths used as compiler *inputs*.

**Decision:** Installed the toolchain itself at `C:\mingw64` (no spaces). The project directory keeps its original path; only the compiler installation needed to move.

**Consequence:** None for the shipped project -- this is purely a local dev-machine detail, recorded here so it isn't rediscovered by surprise later, and reflected in `.toolchain_env.sh` (gitignored) which puts `C:\mingw64\bin` on `PATH` for local builds.

## D3. Python dependency pins (`pyproject.toml`)

**Context:** Spec requires pinned dependency versions for reproducibility. No versions were specified, so pins had to be chosen and verified installable together.

**Decision:** Pinned to versions verified to co-install cleanly in a Python 3.11 venv on this machine: `torch==2.13.0` (CPU wheel locally; Kaggle/CUDA installs are documented separately in `docs/RESULTS.md` once training runs), `numpy==1.26.4`, `OpenEXR==3.4.15` (official bindings, has a prebuilt Windows wheel -- avoids a from-source OpenEXR C++ library build), `scikit-image==0.24.0`, `lpips==0.1.4`, `onnx==1.17.0`, `onnxruntime==1.20.1`, `tensorboard==2.18.0`, `tqdm==4.67.1`, `Pillow==11.0.0`; dev extras `pytest==9.0.2`, `pytest-cov==7.0.0`, `ruff==0.8.4`.

**Consequence:** `pip install -e ".[dev]"` is reproducible on a clean venv. Kaggle notebooks may already ship a compatible torch/CUDA combination; if so the training config documents overriding the torch pin rather than reinstalling.

## D4. Blender EXR Vector-pass convention: assumed, not yet empirically confirmed

**Context:** CLAUDE.md section 3.2 requires writing a test that warps frame t-1 into frame t using the LR vector pass and asserts low error in non-occluded regions for a known camera pan, and explicitly says the convention "must be confirmed empirically, do not assume." Doing that for real requires an actual Blender render, which this environment cannot produce (D0).

**Decision:** Split the risk into two layers. `frameforge.warp.warp` implements one unambiguous, fully-owned convention (backward flow, target->source, pixel units, (dx, dy)) and is exhaustively tested against synthetic data with independently-computed ground truth (`python/tests/test_warp.py`) -- this layer is verified, not assumed. `frameforge.data.exr_io.blender_vector_to_backward_flow` is a thin, currently-identity adapter that converts Blender's raw "Vector" pass into that convention; its docstring and `configs/data.yaml`'s `exr_channel_map` both state plainly that the specific sign/axis/unit choice (matching Blender's documented behavior: RG = motion to previous frame, BA = motion to next frame, in screen-space pixels) is an assumption pending confirmation. `python/tests/test_exr_io.py` runs the exact test the spec asks for, but against a synthetic multilayer-EXR fixture with a known camera pan (`tests/fixtures/generate_fixtures.py`) rather than a real render.

**Consequence:** Everything downstream (dataset loader, model's internal warp, temporal loss) is correct *relative to* the assumed convention, and the one place that assumption lives is a single function with a clear TODO. The action item, to run once Blender is available: render one short real pan sequence, run `test_exr_io.py`'s test against it, and if it fails, fix the sign/axis/scale in `blender_vector_to_backward_flow` (not in `warp.py`, which stays correct either way).

## D5. Scene selection for rendering: deferred to whoever runs the actual render

**Context:** CLAUDE.md section 3.1 asks for 3-4 CC0/CC-BY scenes with scripted flythroughs (pans, dolly, fast rotation, occlusion events), ~40 sequences total, one scene held out entirely for test. Picking and downloading actual Poly Haven assets or Blender demo files needs a Blender install and browsing actual asset catalogs, neither available here (D0).

**Decision:** `configs/render.yaml` ships with the full mechanism (Cycles settings, pass config, a camera-flythrough keyframe DSL with two example flythrough shapes -- a pan+dolly and a fast rotation) but an empty `scenes` list, with instructions and candidate sources (Poly Haven, Blender's official CC-BY demo files) left as comments. `blender/render_sequences.py` is written and syntax-checked against it but not run end to end.

**Consequence:** M1's rendering *stage* is fully scripted and one command away from producing real data, but the real ~40-sequence dataset does not exist in this repo. Whoever runs it next fills in `scenes:` with real `.blend` file paths and license info (adding matching rows to `THIRD_PARTY.md`), then runs `blender -b <file> -P blender/render_sequences.py -- --config configs/render.yaml --scene-name <name> --out data/rendered` once per scene.

## D6. Teacher caching vs. on-the-fly crops: cache covers full frames, live fallback covers crops

**Context:** CLAUDE.md section 4 requires "teacher inference never runs inside the training loop," but section 5.2 trains on random 128px LR crops. A precomputed teacher cache (`frameforge.data.teacher_cache`) necessarily holds full, uncropped sequences -- caching every possible random crop isn't practical. Cropped batches and the cache's full-frame tensors don't line up shape-wise.

**Decision:** `train.py`'s `teacher_cache_dir` path uses the cache whenever a batch's shape matches it (i.e. `crop_size`/`unroll_length` set to `null` in `configs/data.yaml`, training on full frames). When cropping is enabled, it falls back to live teacher inference for that step only, printing a one-time warning, and refuses to start at all if no `--teacher-weights` fallback was given alongside `--teacher-cache-dir` in that configuration (a clear upfront error rather than a crash mid-run). `python/tests/test_train_integration.py` exercises the full experiment matrix (A/B/C/D) through `train()` end to end on the synthetic fixture with `crop_size=null` specifically so the cache path is what's actually tested.

**Consequence:** The caching requirement is met exactly as specified for full-frame training, and degrades honestly (not silently) for the spec's recommended cropped-training setup. Closing this gap properly -- caching stays valid under cropping by sharing crop coordinates between the dataset and the cached tensors -- is a known, scoped follow-up (crop the cached HR/feature tensors using the same random offsets `FrameSequenceDataset` draws, which requires exposing those offsets rather than generating them internally); not done here because it does not change correctness, only which steps hit the (equally correct) live-inference fallback, and the real render/training runs this matters for are themselves still pending (D0).

## D7. ONNX export uses the legacy TorchScript-based exporter, not the dynamo one

**Context:** torch 2.13's `torch.onnx.export` defaults to the newer `torch.export`/dynamo-based exporter, which requires the `onnxscript` package. Installing it pulled in numpy 2.4.6, conflicting with this project's `numpy==1.26.4` pin (D3) and, transitively, the `OpenEXR` bindings' resolved dependency set.

**Decision:** Pass `dynamo=False` to `torch.onnx.export` (export_onnx.py), forcing the legacy TorchScript-based tracer. It handles `grid_sample` -> ONNX `GridSample` correctly at opset 17 with no extra dependency. Confirmed via `python/tests/test_onnx_parity.py`: PyTorch vs ONNX Runtime max abs error is far under the 1e-4 spec threshold across all four model variants, and the exported graph correctly runs at a different H/W than it was exported with (dynamic axes).

**Consequence:** A future numpy 2.x upgrade (e.g. to unblock some other dependency) would need to re-evaluate this choice, since the dynamo exporter is where PyTorch's ONNX export effort is headed long-term (the legacy path prints a deprecation warning). Not a concern for the current pinned dependency set.

## D8. ASan/UBSan/TSan cannot be verified locally on this Windows/MinGW toolchain

**Context:** CLAUDE.md section 8 requires `cpp-asan-ubsan` and `cpp-tsan` CI jobs (configured in `.github/workflows/ci.yml` to run on `ubuntu-latest`, where GCC ships full sanitizer runtimes). Tried building locally with `-DFRAMEFORGE_ENABLE_ASAN=ON` using this project's MinGW-w64 GCC 16.2 toolchain (D1): linking fails with `cannot find -lasan` / `-lubsan`. Checked: this WinLibs distribution ships the sanitizer *headers* (`asan.h`, `ubsan.h`, `tsan.h`, for annotations) but not the runtime libraries themselves -- a known, common gap in prebuilt MinGW-w64 toolchains, since sanitizer runtime support for the MinGW target needs extra build configuration most distributors skip (MSVC's own `/fsanitize=address`, added in VS2019, is the more common Windows-native path instead, but this project's toolchain is GCC for Linux-CI parity, not MSVC).

**Decision:** Do not chase a working local ASan/TSan setup on Windows -- treat CI's Ubuntu `cpp-asan-ubsan`/`cpp-tsan` jobs as the authoritative verification, consistent with D0's general "CI is the authoritative Linux-side check" pattern. Locally, the concurrency tests (`queue_test.cpp`'s contention/stress cases) are still real and exercised under normal (non-instrumented) execution -- see the full `ctest` run, 28/28 passing, including the multi-thousand-item multi-producer/consumer stress tests.

**Consequence:** This repo's own dev loop cannot catch a data race or memory error before CI does. Not a correctness gap in the code itself, but a real verification gap worth being upfront about: the queue/pipeline code has NOT been sanitizer-verified as of this commit, only functionally tested. First priority once this runs on GitHub Actions: confirm the `cpp-asan-ubsan` and `cpp-tsan` jobs are actually green, not just configured.

## D9. C++ engine build: MinGW/ONNX Runtime linking, tinyexr vendoring, and the pipeline's real overlap boundary

**Context:** M4/M5 needed the actual C++ engine (ONNX Runtime C++ API, tinyexr, the concurrent pipeline) built and tested for real, not just authored. Three real problems came up building it with this project's MinGW-w64 GCC 16.2 toolchain (D1):

1. **ONNX Runtime's C API header hard-codes `_stdcall`** (an MSVC-only spelling) under `#ifdef _WIN32`, which GCC doesn't define, breaking every function-pointer declaration in `onnxruntime_c_api.h`. Fix: `#define _stdcall` before including it (`engine/include/frameforge/ort_session.hpp`) -- x64 has one calling convention, so this is a no-op, not a behavior change. Confirmed by an actual compile+link+run round trip against the official prebuilt `onnxruntime.dll` (session creation, tensor I/O, `Run()`) before writing any engine code around it.
2. **tinyexr's `release` branch tip is not self-contained**: it `#include`s `exr_reader.hh`, which isn't in a plain `tinyexr.h` download from that branch. Switched to the stable `v1.0.13` tag, vendored its actual companion files (`exr_reader.hh`, `streamreader.hh`) alongside `tinyexr.h`, and its bundled `miniz` (`deps/miniz/miniz.h` + `.c`) since `TINYEXR_USE_MINIZ` is on by default and expects `<miniz.h>` on the include path.
3. **miniz.c is genuine C, not C++-compilable as-is**: it has a `static const mz_uint s_tdefl_num_probes[11];` tentative declaration followed by its real definition elsewhere in the file -- valid C, a redefinition error in C++. First tried forcing it to compile as C++ (`set_source_files_properties(... LANGUAGE CXX)`); that failed for this exact reason. Fixed properly by adding C to `project(... LANGUAGES CXX C)` and letting it compile as C.

**Decision (pipeline overlap boundary):** CLAUDE.md 7.3 asks for a reader/inference/writer pipeline connected by bounded queues. Per-sequence depth normalization (CLAUDE.md 3.3, matching `frameforge.colorspace.normalize_depth`) needs every frame's depth range before ANY frame can be normalized, so the reader stage cannot start streaming frames to the inference stage until it has decoded the whole sequence -- reading and inference genuinely cannot overlap under this design, not as a missed optimization but as a direct consequence of per-sequence (not per-frame) normalization. What DOES overlap for real, and is implemented and tested as such (`engine/tests/pipeline_test.cpp`, byte-identical output to the sequential path): inference and writing, since PNG-encoding frame *t*'s output has no dependency on frame *t+1*'s inference.

**Decision (queue choice):** `engine/bench/queue_bench.cpp` measured `SpscRingBuffer` at ~1.2x (1000 items) to ~5.6x (100,000 items) the throughput of `BlockingQueue` on this hardware (see docs/RESULTS.md for the numbers and hardware/software versions). `SpscRingBuffer` is the pipeline's default per CLAUDE.md 7.3's "keep the better one as default."

**Consequence:** The engine is genuinely built, linked, and tested on this machine (31/31 GoogleTest cases passing, including engine-output-vs-PyTorch-reference and tiled-vs-untiled equivalence), not just written against documentation. The honest architectural limit above (reader/inference can't overlap) is recorded here and in `docs/ARCHITECTURE.md` rather than glossed over.
