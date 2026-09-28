#include "frameforge/png_io.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

namespace frameforge {

void WritePng(const std::string& path, const Tensor& rgb_0_to_1) {
  if (rgb_0_to_1.n() != 1 || rgb_0_to_1.c() != 3) {
    throw std::invalid_argument("WritePng expects a (1, 3, H, W) tensor");
  }
  const int h = rgb_0_to_1.h();
  const int w = rgb_0_to_1.w();
  std::vector<unsigned char> pixels(static_cast<size_t>(h) * w * 3);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      for (int c = 0; c < 3; ++c) {
        float v = rgb_0_to_1.at(0, c, y, x);
        v = std::clamp(v, 0.0f, 1.0f);
        pixels[(static_cast<size_t>(y) * w + x) * 3 + c] =
            static_cast<unsigned char>(v * 255.0f + 0.5f);
      }
    }
  }
  if (!stbi_write_png(path.c_str(), w, h, 3, pixels.data(), w * 3)) {
    throw std::runtime_error("failed to write PNG: " + path);
  }
}

Tensor ReadPngAsFloat(const std::string& path) {
  int w, h, channels;
  unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &channels, 3);
  if (pixels == nullptr) {
    throw std::runtime_error("failed to load PNG: " + path);
  }
  Tensor out(1, 3, h, w);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      for (int c = 0; c < 3; ++c) {
        out.at(0, c, y, x) = pixels[(static_cast<size_t>(y) * w + x) * 3 + c] / 255.0f;
      }
    }
  }
  stbi_image_free(pixels);
  return out;
}

}  // namespace frameforge
