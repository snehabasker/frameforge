// PNG write via stb_image_write (vendored at engine/include/stb/, public
// domain -- see THIRD_PARTY.md). `Tensor` here is expected in [0, 1]
// (tonemapped sRGB), NCHW, batch size 1.
#pragma once

#include <string>

#include "frameforge/tensor.hpp"

namespace frameforge {

void WritePng(const std::string& path, const Tensor& rgb_0_to_1);

// Reads a PNG (any format stb_image supports) as [0, 1] float, NCHW (1, 3, H, W).
Tensor ReadPngAsFloat(const std::string& path);

}  // namespace frameforge
