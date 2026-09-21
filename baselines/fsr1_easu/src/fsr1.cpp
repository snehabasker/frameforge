// See include/fsr1/fsr1.hpp for the framing/license/deviations note.
// Ported from AMD's ffx_fsr1.h (MIT), vendored at ../vendor/ffx_fsr1.h.
// Comments below reference the original function/variable names so this
// stays traceable against the upstream source line-by-line.

#include "fsr1/fsr1.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace fsr1 {

namespace {

constexpr float kTiny = 1e-8f;

// Exact reciprocal with a zero-safe fallback, replacing the GPU path's
// approximate reciprocal intrinsics (see hpp's "known deviations").
float safe_rcp(float x) { return std::abs(x) > kTiny ? 1.0f / x : 0.0f; }

float saturate(float x) { return std::clamp(x, 0.0f, 1.0f); }

int clampi(int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); }

struct Vec3 {
  float r = 0, g = 0, b = 0;
  Vec3 operator+(const Vec3& o) const { return {r + o.r, g + o.g, b + o.b}; }
  Vec3 operator*(float s) const { return {r * s, g * s, b * s}; }
};

Vec3 min3(const Vec3& a, const Vec3& b) {
  return {std::min(a.r, b.r), std::min(a.g, b.g), std::min(a.b, b.b)};
}
Vec3 max3(const Vec3& a, const Vec3& b) {
  return {std::max(a.r, b.r), std::max(a.g, b.g), std::max(a.b, b.b)};
}

// Clamped-edge tap fetch (FsrEasuRF/GF/BF's job on the GPU, via a clamped
// texture sampler).
Vec3 tap(const Image& src, int x, int y) {
  x = clampi(x, 0, src.width - 1);
  y = clampi(y, 0, src.height - 1);
  return {src.at(x, y, 0), src.at(x, y, 1), src.at(x, y, 2)};
}

// Simple multi-channel approximate luma times 2 (matches ffx_fsr1.h exactly).
float luma2(const Vec3& c) { return c.b * 0.5f + (c.r * 0.5f + c.g); }

// FsrEasuSetF: accumulate gradient direction and length from one of the
// 2x2 cell's four corners.
void easu_set(float& dir_x, float& dir_y, float& len, float ppx, float ppy, bool bi_s, bool bi_t,
              bool bi_u, bool bi_v, float lA, float lB, float lC, float lD, float lE) {
  float w = 0.0f;
  if (bi_s) w = (1.0f - ppx) * (1.0f - ppy);
  if (bi_t) w = ppx * (1.0f - ppy);
  if (bi_u) w = (1.0f - ppx) * ppy;
  if (bi_v) w = ppx * ppy;

  float dc = lD - lC;
  float cb = lC - lB;
  float len_x = safe_rcp(std::max(std::abs(dc), std::abs(cb)));
  float dir_x_term = lD - lB;
  dir_x += dir_x_term * w;
  len_x = saturate(std::abs(dir_x_term) * len_x);
  len_x *= len_x;
  len += len_x * w;

  float ec = lE - lC;
  float ca = lC - lA;
  float len_y = safe_rcp(std::max(std::abs(ec), std::abs(ca)));
  float dir_y_term = lE - lA;
  dir_y += dir_y_term * w;
  len_y = saturate(std::abs(dir_y_term) * len_y);
  len_y *= len_y;
  len += len_y * w;
}

// FsrEasuTapF: accumulate one of the 12 kernel taps into the weighted sum.
void easu_tap(Vec3& acc_color, float& acc_weight, float off_x, float off_y, float dir_x,
              float dir_y, float len_x, float len_y, float lob, float clp, const Vec3& c) {
  float vx = off_x * dir_x + off_y * dir_y;
  float vy = off_x * (-dir_y) + off_y * dir_x;
  vx *= len_x;
  vy *= len_y;
  float d2 = vx * vx + vy * vy;
  d2 = std::min(d2, clp);

  float wb = (2.0f / 5.0f) * d2 - 1.0f;
  float wa = lob * d2 - 1.0f;
  wb *= wb;
  wa *= wa;
  wb = (25.0f / 16.0f) * wb - (25.0f / 16.0f - 1.0f);
  float w = wb * wa;

  acc_color = acc_color + c * w;
  acc_weight += w;
}

}  // namespace

Image easu_upscale(const Image& src, int out_width, int out_height) {
  Image dst(out_width, out_height);
  const float scale_x = static_cast<float>(src.width) / static_cast<float>(out_width);
  const float scale_y = static_cast<float>(src.height) / static_cast<float>(out_height);

  for (int oy = 0; oy < out_height; ++oy) {
    for (int ox = 0; ox < out_width; ++ox) {
      // Position of 'f' (FsrEasuCon's con0 + the pp/fp split in FsrEasuF).
      float px = (static_cast<float>(ox) + 0.5f) * scale_x - 0.5f;
      float py = (static_cast<float>(oy) + 0.5f) * scale_y - 0.5f;
      float fx = std::floor(px);
      float fy = std::floor(py);
      float ppx = px - fx;
      float ppy = py - fy;
      int ifx = static_cast<int>(fx);
      int ify = static_cast<int>(fy);

      // 12-tap neighborhood, named exactly as in the ffx_fsr1.h diagram:
      //    b c
      //  e f g h
      //  i j k l
      //    n o
      Vec3 b = tap(src, ifx + 0, ify - 1);
      Vec3 c = tap(src, ifx + 1, ify - 1);
      Vec3 e = tap(src, ifx - 1, ify + 0);
      Vec3 f = tap(src, ifx + 0, ify + 0);
      Vec3 g = tap(src, ifx + 1, ify + 0);
      Vec3 h = tap(src, ifx + 2, ify + 0);
      Vec3 i = tap(src, ifx - 1, ify + 1);
      Vec3 j = tap(src, ifx + 0, ify + 1);
      Vec3 k = tap(src, ifx + 1, ify + 1);
      Vec3 l = tap(src, ifx + 2, ify + 1);
      Vec3 n = tap(src, ifx + 0, ify + 2);
      Vec3 o = tap(src, ifx + 1, ify + 2);

      float bL = luma2(b), cL = luma2(c), eL = luma2(e), fL = luma2(f);
      float gL = luma2(g), hL = luma2(h), iL = luma2(i), jL = luma2(j);
      float kL = luma2(k), lL = luma2(l), nL = luma2(n), oL = luma2(o);

      float dir_x = 0.0f, dir_y = 0.0f, len = 0.0f;
      easu_set(dir_x, dir_y, len, ppx, ppy, true, false, false, false, bL, eL, fL, gL, jL);
      easu_set(dir_x, dir_y, len, ppx, ppy, false, true, false, false, cL, fL, gL, hL, kL);
      easu_set(dir_x, dir_y, len, ppx, ppy, false, false, true, false, fL, iL, jL, kL, nL);
      easu_set(dir_x, dir_y, len, ppx, ppy, false, false, false, true, gL, jL, kL, lL, oL);

      float dir2 = dir_x * dir_x + dir_y * dir_y;
      bool zero_dir = dir2 < (1.0f / 32768.0f);
      float dir_rcp_sqrt = zero_dir ? 1.0f : 1.0f / std::sqrt(dir2);
      if (zero_dir) dir_x = 1.0f;
      dir_x *= dir_rcp_sqrt;
      dir_y *= dir_rcp_sqrt;

      len = len * 0.5f;
      len *= len;

      float stretch = (dir_x * dir_x + dir_y * dir_y) *
                       safe_rcp(std::max(std::abs(dir_x), std::abs(dir_y)));
      float len_x = 1.0f + (stretch - 1.0f) * len;
      float len_y = 1.0f + (-0.5f) * len;
      float lob = 0.5f + ((1.0f / 4.0f - 0.04f) - 0.5f) * len;
      float clp = safe_rcp(lob);

      Vec3 min4 = min3(min3(min3(f, k), j), g);
      Vec3 max4 = max3(max3(max3(f, k), j), g);

      Vec3 acc_color{0, 0, 0};
      float acc_weight = 0.0f;
      easu_tap(acc_color, acc_weight, 0.0f - ppx, -1.0f - ppy, dir_x, dir_y, len_x, len_y, lob, clp, b);
      easu_tap(acc_color, acc_weight, 1.0f - ppx, -1.0f - ppy, dir_x, dir_y, len_x, len_y, lob, clp, c);
      easu_tap(acc_color, acc_weight, -1.0f - ppx, 1.0f - ppy, dir_x, dir_y, len_x, len_y, lob, clp, i);
      easu_tap(acc_color, acc_weight, 0.0f - ppx, 1.0f - ppy, dir_x, dir_y, len_x, len_y, lob, clp, j);
      easu_tap(acc_color, acc_weight, 0.0f - ppx, 0.0f - ppy, dir_x, dir_y, len_x, len_y, lob, clp, f);
      easu_tap(acc_color, acc_weight, -1.0f - ppx, 0.0f - ppy, dir_x, dir_y, len_x, len_y, lob, clp, e);
      easu_tap(acc_color, acc_weight, 1.0f - ppx, 1.0f - ppy, dir_x, dir_y, len_x, len_y, lob, clp, k);
      easu_tap(acc_color, acc_weight, 2.0f - ppx, 1.0f - ppy, dir_x, dir_y, len_x, len_y, lob, clp, l);
      easu_tap(acc_color, acc_weight, 2.0f - ppx, 0.0f - ppy, dir_x, dir_y, len_x, len_y, lob, clp, h);
      easu_tap(acc_color, acc_weight, 1.0f - ppx, 0.0f - ppy, dir_x, dir_y, len_x, len_y, lob, clp, g);
      easu_tap(acc_color, acc_weight, 1.0f - ppx, 2.0f - ppy, dir_x, dir_y, len_x, len_y, lob, clp, o);
      easu_tap(acc_color, acc_weight, 0.0f - ppx, 2.0f - ppy, dir_x, dir_y, len_x, len_y, lob, clp, n);

      Vec3 resolved = acc_color * safe_rcp(acc_weight);
      Vec3 pix = max3(min4, min3(max4, resolved));

      dst.at(ox, oy, 0) = pix.r;
      dst.at(ox, oy, 1) = pix.g;
      dst.at(ox, oy, 2) = pix.b;
    }
  }
  return dst;
}

Image rcas_sharpen(const Image& src, float sharpness) {
  Image dst(src.width, src.height);
  const float con = std::exp2(-sharpness);  // FsrRcasCon
  constexpr float kRcasLimit = 0.25f - 1.0f / 16.0f;

  for (int y = 0; y < src.height; ++y) {
    for (int x = 0; x < src.width; ++x) {
      // 3x3 cross neighborhood:
      //    b
      //  d e f
      //    h
      Vec3 b = tap(src, x, y - 1);
      Vec3 d = tap(src, x - 1, y);
      Vec3 e = tap(src, x, y);
      Vec3 f = tap(src, x + 1, y);
      Vec3 h = tap(src, x, y + 1);

      Vec3 min4{std::min({b.r, d.r, f.r, h.r}), std::min({b.g, d.g, f.g, h.g}),
                std::min({b.b, d.b, f.b, h.b})};
      Vec3 max4{std::max({b.r, d.r, f.r, h.r}), std::max({b.g, d.g, f.g, h.g}),
                std::max({b.b, d.b, f.b, h.b})};

      auto lobe_channel = [&](float mn4, float mx4, float e_c) {
        float hit_min = std::min(mn4, e_c) * safe_rcp(4.0f * mx4);
        float hit_max = (1.0f - std::max(mx4, e_c)) * safe_rcp(4.0f * mn4 - 4.0f);
        return std::max(-hit_min, hit_max);
      };
      float lobe_r = lobe_channel(min4.r, max4.r, e.r);
      float lobe_g = lobe_channel(min4.g, max4.g, e.g);
      float lobe_b = lobe_channel(min4.b, max4.b, e.b);
      float lobe = std::max(-kRcasLimit, std::min(std::max({lobe_r, lobe_g, lobe_b}), 0.0f)) * con;

      float rcp_l = 1.0f / (4.0f * lobe + 1.0f);  // always in (0, 1], safe exact
      Vec3 out{
          (lobe * b.r + lobe * d.r + lobe * h.r + lobe * f.r + e.r) * rcp_l,
          (lobe * b.g + lobe * d.g + lobe * h.g + lobe * f.g + e.g) * rcp_l,
          (lobe * b.b + lobe * d.b + lobe * h.b + lobe * f.b + e.b) * rcp_l,
      };
      dst.at(x, y, 0) = out.r;
      dst.at(x, y, 1) = out.g;
      dst.at(x, y, 2) = out.b;
    }
  }
  return dst;
}

}  // namespace fsr1
