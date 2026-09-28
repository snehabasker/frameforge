// Fixed tonemap + sRGB encode, mirroring python/frameforge/colorspace.py
// EXACTLY (same formulas) -- CLAUDE.md 3.3: "Use the same transform
// everywhere." Any change here must be mirrored there, and vice versa;
// engine/tests/tonemap_test.cpp cross-checks fixed values against ones
// computed by the Python module (see that test for the exact numbers).
#pragma once

#include <cmath>

namespace frameforge {

inline float ReinhardTonemap(float linear_hdr, float white_point = -1.0f) {
  if (white_point < 0.0f) {
    return linear_hdr / (1.0f + linear_hdr);
  }
  float lwhite2 = white_point * white_point;
  return (linear_hdr * (1.0f + linear_hdr / lwhite2)) / (1.0f + linear_hdr);
}

inline float LinearToSrgb(float linear) {
  if (linear < 0.0f) linear = 0.0f;
  if (linear <= 0.0031308f) {
    return linear * 12.92f;
  }
  constexpr float a = 0.055f;
  float high = (1.0f + a) * std::pow(linear, 1.0f / 2.4f) - a;
  return high > 1.0f ? 1.0f : (high < 0.0f ? 0.0f : high);
}

inline float SrgbToLinear(float srgb) {
  if (srgb < 0.0f) srgb = 0.0f;
  if (srgb > 1.0f) srgb = 1.0f;
  if (srgb <= 0.04045f) {
    return srgb / 12.92f;
  }
  constexpr float a = 0.055f;
  return std::pow((srgb + a) / (1.0f + a), 2.4f);
}

// The one fixed transform: linear HDR -> tonemapped sRGB in [0, 1].
inline float TonemapAndEncode(float linear_hdr, float white_point = -1.0f) {
  return LinearToSrgb(ReinhardTonemap(linear_hdr, white_point));
}

}  // namespace frameforge
