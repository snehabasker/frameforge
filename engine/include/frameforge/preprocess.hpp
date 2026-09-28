// Shared preprocessing steps (tonemap, depth normalization, bilinear
// upsample) used by both the sequential pipeline (sequence_runner.cpp)
// and the threaded pipeline (pipeline.cpp) -- exactly one implementation
// of each, so they can never silently drift apart.
#pragma once

#include <vector>

#include "frameforge/tensor.hpp"

namespace frameforge {

Tensor ApplyTonemap(const Tensor& linear_hdr);

// Matches frameforge.colorspace.normalize_depth exactly: per-sequence
// min/max over finite values; non-finite (e.g. sky/background) clamped to
// the finite max before normalizing.
void NormalizeDepthInPlace(std::vector<Tensor>& depths);

Tensor BilinearUpsample2x(const Tensor& lr);

}  // namespace frameforge
