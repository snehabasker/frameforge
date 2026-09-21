# Third-party assets and code

Every non-original asset or code fragment used in this repository, with its license. Nothing here may be anything other than CC0 / CC-BY / MIT / Apache-2.0 / BSD, per project rules.

## Code

| Component | Source | License | Used for |
|---|---|---|---|
| SwinIR (classical SR, x2) | https://github.com/JingyunLiang/SwinIR | Apache-2.0 | Frozen teacher model. Code vendored verbatim at `python/frameforge/models/_swinir_vendor/network_swinir.py`. Pretrained weights: `001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth` from the repo's `v0.0` GitHub release, fetched + SHA256-verified by `scripts/download_pretrained.py` (never committed, per CLAUDE.md 11) |
| timm (`DropPath`, `to_2tuple`, `trunc_normal_`) | https://github.com/huggingface/pytorch-image-models | Apache-2.0 | Runtime dependency of the vendored SwinIR code above (its own `import` statement); added to `pyproject.toml` |
| FidelityFX FSR 1 (`ffx_fsr1.h`) | https://github.com/GPUOpen-Effects/FidelityFX-FSR | MIT | Reference for the CPU EASU/RCAS port in `baselines/fsr1_easu/` (quality comparison only, not a performance baseline -- see `baselines/fsr1_easu/README.md`) |
| tinyexr | https://github.com/syoyo/tinyexr | BSD-3-Clause | EXR I/O in the C++ engine |
| stb_image / stb_image_write | https://github.com/nothings/stb | Public domain / MIT (dual) | PNG I/O in the C++ engine |
| GoogleTest | https://github.com/google/googletest | BSD-3-Clause | C++ unit tests |
| Google Benchmark | https://github.com/google/benchmark | Apache-2.0 | C++ microbenchmarks |
| ONNX Runtime (prebuilt binary) | https://github.com/microsoft/onnxruntime | MIT | C++ inference engine |
| LPIPS (`lpips` PyPI package) | https://github.com/richzhang/PerceptualSimilarity | BSD-2-Clause | Perceptual metric in evaluation |
| FlameGraph scripts | https://github.com/brendangregg/FlameGraph | CDDL-1.0 (scripts only, not linked into any shipped binary) | `scripts/flamegraph.sh` profiling output |

## Data / rendered assets

Populated in M1 once specific Poly Haven / Blender demo scenes are selected. Each entry will record: asset name, source URL, license (CC0 or CC-BY, with attribution text if CC-BY), and which sequence it produced.

| Asset | Source | License | Attribution required |
|---|---|---|---|
| _pending M1_ | | | |
