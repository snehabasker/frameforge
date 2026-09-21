// CPU reference port of AMD FidelityFX Super Resolution 1.0 (FSR1):
// EASU (Edge-Adaptive Spatial Upsampling) + RCAS (Robust Contrast-Adaptive
// Sharpening), faithfully following the algorithm in the official,
// MIT-licensed `ffx_fsr1.h` (vendored at baselines/fsr1_easu/vendor/,
// see LICENSE.txt there and THIRD_PARTY.md).
//
// FRAMING (CLAUDE.md 6.1): this port exists ONLY for offline quality
// comparison against FrameForge's output. Its CPU speed must never be
// reported as "FSR1 performance" -- the real FSR1 runs as a GPU compute
// shader and is dramatically faster there. See baselines/fsr1_easu/README.md.
//
// Known deviations from the original (documented, not hidden):
//   - The GPU shader uses fast approximate reciprocal/rsqrt intrinsics
//     (APrxLoRcpF1, APrxLoRsqF1, APrxMedRcpF1) for shader-speed reasons.
//     This CPU port uses exact 1/x and 1/sqrt(x) throughout -- strictly
//     more accurate, deterministic, and irrelevant to a reference
//     (non-realtime) CPU implementation. This is the one deliberate
//     algorithmic substitution; everything else (the 12-tap EASU kernel,
//     direction/length accumulation, lobe shaping, RCAS's 3x3 neighborhood
//     and noise-aware clamping) is a direct translation.
//   - No packed-FP16 path (GPU-only optimization, meaningless on CPU).
//   - Edge pixels use clamped (nearest-edge) sampling for out-of-bounds
//     taps, matching a clamped texture sampler (the GPU path's implicit
//     behavior via its texture unit's clamp addressing mode).
#pragma once

#include <cstdint>
#include <vector>

namespace fsr1 {

// Simple planar-interleaved RGB float image, values expected in [0, 1]
// (tonemapped sRGB, same convention as the rest of the project).
struct Image {
  int width = 0;
  int height = 0;
  std::vector<float> data;  // size = width * height * 3, row-major, RGB-interleaved

  Image() = default;
  Image(int w, int h) : width(w), height(h), data(static_cast<size_t>(w) * h * 3, 0.0f) {}

  float& at(int x, int y, int c) {
    return data[(static_cast<size_t>(y) * width + x) * 3 + c];
  }
  float at(int x, int y, int c) const {
    return data[(static_cast<size_t>(y) * width + x) * 3 + c];
  }
};

// Edge-Adaptive Spatial Upsampling: `src` (width x height) -> a new image
// of size (outWidth x outHeight). Faithful port of FsrEasuCon + FsrEasuF.
Image easu_upscale(const Image& src, int out_width, int out_height);

// Robust Contrast-Adaptive Sharpening, applied in place at `src`'s own
// resolution (typically called on EASU's output). `sharpness` matches the
// original's convention: 0.0 = maximum sharpness, larger N = N stops
// (halvings) less sharp. Faithful port of FsrRcasCon + FsrRcasF.
Image rcas_sharpen(const Image& src, float sharpness);

}  // namespace fsr1
