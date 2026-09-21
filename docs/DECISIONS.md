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
