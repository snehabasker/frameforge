// Minimal owning NCHW float tensor. RAII throughout (CLAUDE.md 7.3): no
// raw new/delete, backed by std::vector.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace frameforge {

class Tensor {
 public:
  Tensor() = default;
  Tensor(int n, int c, int h, int w)
      : n_(n), c_(c), h_(h), w_(w), data_(static_cast<size_t>(n) * c * h * w, 0.0f) {}

  int n() const { return n_; }
  int c() const { return c_; }
  int h() const { return h_; }
  int w() const { return w_; }
  size_t size() const { return data_.size(); }

  float* data() { return data_.data(); }
  const float* data() const { return data_.data(); }

  float& at(int ni, int ci, int hi, int wi) {
    return data_[Index(ni, ci, hi, wi)];
  }
  float at(int ni, int ci, int hi, int wi) const { return data_[Index(ni, ci, hi, wi)]; }

  std::vector<int64_t> shape() const { return {n_, c_, h_, w_}; }

 private:
  size_t Index(int ni, int ci, int hi, int wi) const {
    return (((static_cast<size_t>(ni) * c_ + ci) * h_ + hi) * w_ + wi);
  }

  int n_ = 0, c_ = 0, h_ = 0, w_ = 0;
  std::vector<float> data_;
};

// HWC (row-major, interleaved channels -- e.g. straight off an image
// decoder) <-> NCHW (planar, batch-first -- what the ONNX graph expects)
// conversions. CLAUDE.md 7.4 explicitly wants these tested.
inline Tensor HwcToNchw(const std::vector<float>& hwc, int h, int w, int c) {
  Tensor out(1, c, h, w);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      for (int ch = 0; ch < c; ++ch) {
        out.at(0, ch, y, x) = hwc[(static_cast<size_t>(y) * w + x) * c + ch];
      }
    }
  }
  return out;
}

inline std::vector<float> NchwToHwc(const Tensor& t) {
  if (t.n() != 1) {
    throw std::invalid_argument("NchwToHwc: only batch size 1 supported, got n=" +
                                 std::to_string(t.n()));
  }
  std::vector<float> out(static_cast<size_t>(t.h()) * t.w() * t.c());
  for (int y = 0; y < t.h(); ++y) {
    for (int x = 0; x < t.w(); ++x) {
      for (int ch = 0; ch < t.c(); ++ch) {
        out[(static_cast<size_t>(y) * t.w() + x) * t.c() + ch] = t.at(0, ch, y, x);
      }
    }
  }
  return out;
}

}  // namespace frameforge
