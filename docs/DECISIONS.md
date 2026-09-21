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
