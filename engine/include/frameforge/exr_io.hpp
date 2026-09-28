// Multilayer EXR reading via tinyexr (vendored at engine/include/tinyexr/,
// BSD-3-Clause -- see THIRD_PARTY.md). Mirrors the channel-name convention
// python/frameforge/data/exr_io.py uses (configs/data.yaml's
// exr_channel_map): "<ViewLayer>.<Pass>.<Channel>".
#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "frameforge/tensor.hpp"

namespace frameforge {

// One channel's pixel data, always as float32 regardless of how it was
// stored (half/float) in the file -- tinyexr does this conversion for us
// when requested_pixel_types is set accordingly.
struct ExrFrame {
  int width = 0;
  int height = 0;
  std::map<std::string, std::vector<float>> channels;  // full dotted name -> HxW floats

  const std::vector<float>& Channel(const std::string& name) const {
    auto it = channels.find(name);
    if (it == channels.end()) {
      throw std::runtime_error("EXR file missing channel: " + name);
    }
    return it->second;
  }
};

// Throws std::runtime_error on any failure (missing file, parse error,
// unsupported format), with tinyexr's own error string included.
ExrFrame ReadExrFrame(const std::string& path);

struct ChannelMap {
  std::string view_layer = "ViewLayer";
  std::string combined_pass = "Combined";
  std::string depth_pass = "Depth";
  std::string vector_pass = "Vector";
  std::string vector_to_previous_x = "X";
  std::string vector_to_previous_y = "Y";
};

// Extracts (combined RGB, depth, vector-to-previous) as Tensors, at (1, C,
// H, W). Depth is NOT normalized here (that needs the whole sequence's
// min/max, done one level up, matching frameforge.colorspace.normalize_depth's
// per-sequence contract) -- this function only does the raw EXR -> tensor
// step.
struct RawFramePasses {
  Tensor combined;  // (1, 3, H, W), linear HDR
  Tensor depth;     // (1, 1, H, W), raw Z
  Tensor vector_to_previous;  // (1, 2, H, W), raw Blender units
};

RawFramePasses LoadFramePasses(const std::string& path, const ChannelMap& map);

}  // namespace frameforge
