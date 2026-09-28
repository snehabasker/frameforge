#include "frameforge/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "frameforge/tonemap.hpp"

namespace frameforge {

Tensor ApplyTonemap(const Tensor& linear_hdr) {
  Tensor out(linear_hdr.n(), linear_hdr.c(), linear_hdr.h(), linear_hdr.w());
  for (size_t i = 0; i < linear_hdr.size(); ++i) {
    out.data()[i] = TonemapAndEncode(linear_hdr.data()[i]);
  }
  return out;
}

void NormalizeDepthInPlace(std::vector<Tensor>& depths) {
  float d_min = std::numeric_limits<float>::infinity();
  float d_max = -std::numeric_limits<float>::infinity();
  for (auto& d : depths) {
    for (size_t i = 0; i < d.size(); ++i) {
      float v = d.data()[i];
      if (std::isfinite(v)) {
        d_min = std::min(d_min, v);
        d_max = std::max(d_max, v);
      }
    }
  }
  if (!std::isfinite(d_min)) {
    for (auto& d : depths) std::fill(d.data(), d.data() + d.size(), 0.0f);
    return;
  }
  const float eps = 1e-6f;
  for (auto& d : depths) {
    for (size_t i = 0; i < d.size(); ++i) {
      float v = d.data()[i];
      if (!std::isfinite(v)) v = d_max;
      d.data()[i] = (v - d_min) / (d_max - d_min + eps);
    }
  }
}

Tensor BilinearUpsample2x(const Tensor& lr) {
  Tensor out(1, lr.c(), lr.h() * 2, lr.w() * 2);
  for (int c = 0; c < lr.c(); ++c) {
    for (int y = 0; y < out.h(); ++y) {
      float sy = (y + 0.5f) * 0.5f - 0.5f;
      int y0 = static_cast<int>(std::floor(sy));
      float fy = sy - y0;
      int y0c = std::clamp(y0, 0, lr.h() - 1);
      int y1c = std::clamp(y0 + 1, 0, lr.h() - 1);
      for (int x = 0; x < out.w(); ++x) {
        float sx = (x + 0.5f) * 0.5f - 0.5f;
        int x0 = static_cast<int>(std::floor(sx));
        float fx = sx - x0;
        int x0c = std::clamp(x0, 0, lr.w() - 1);
        int x1c = std::clamp(x0 + 1, 0, lr.w() - 1);
        float v00 = lr.at(0, c, y0c, x0c);
        float v01 = lr.at(0, c, y0c, x1c);
        float v10 = lr.at(0, c, y1c, x0c);
        float v11 = lr.at(0, c, y1c, x1c);
        float top = v00 + (v01 - v00) * fx;
        float bot = v10 + (v11 - v10) * fx;
        out.at(0, c, y, x) = top + (bot - top) * fy;
      }
    }
  }
  return out;
}

}  // namespace frameforge
