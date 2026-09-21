// CLI for the FSR1 CPU reference port (CLAUDE.md 6.1: "Expose it via a
// CLI (and pybind11 or a subprocess for evaluation)" -- this project uses
// the subprocess route: evaluate.py shells out to this binary).
//
// Usage:
//   fsr1_cli --in in.png --out out.png --width W --height H [--sharpness S]
//
// Reads any stb_image-supported format, writes PNG. `--sharpness` is
// optional; when given, RCAS runs after EASU (0.0 = max sharpness).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "fsr1/fsr1.hpp"

namespace {

fsr1::Image LoadPng(const std::string& path) {
  int w, h, channels;
  unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &channels, 3);
  if (pixels == nullptr) {
    std::fprintf(stderr, "failed to load image: %s\n", path.c_str());
    std::exit(1);
  }
  fsr1::Image img(w, h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      for (int c = 0; c < 3; ++c) {
        img.at(x, y, c) = pixels[(y * w + x) * 3 + c] / 255.0f;
      }
    }
  }
  stbi_image_free(pixels);
  return img;
}

void SavePng(const std::string& path, const fsr1::Image& img) {
  std::vector<unsigned char> pixels(static_cast<size_t>(img.width) * img.height * 3);
  for (int y = 0; y < img.height; ++y) {
    for (int x = 0; x < img.width; ++x) {
      for (int c = 0; c < 3; ++c) {
        float v = img.at(x, y, c);
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        pixels[(y * img.width + x) * 3 + c] = static_cast<unsigned char>(v * 255.0f + 0.5f);
      }
    }
  }
  if (!stbi_write_png(path.c_str(), img.width, img.height, 3, pixels.data(), img.width * 3)) {
    std::fprintf(stderr, "failed to write image: %s\n", path.c_str());
    std::exit(1);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string in_path, out_path;
  int out_width = 0, out_height = 0;
  bool have_sharpness = false;
  float sharpness = 0.0f;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", arg.c_str());
        std::exit(1);
      }
      return argv[++i];
    };
    if (arg == "--in") {
      in_path = next();
    } else if (arg == "--out") {
      out_path = next();
    } else if (arg == "--width") {
      out_width = std::atoi(next().c_str());
    } else if (arg == "--height") {
      out_height = std::atoi(next().c_str());
    } else if (arg == "--sharpness") {
      sharpness = static_cast<float>(std::atof(next().c_str()));
      have_sharpness = true;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 1;
    }
  }

  if (in_path.empty() || out_path.empty() || out_width <= 0 || out_height <= 0) {
    std::fprintf(stderr,
                 "usage: fsr1_cli --in in.png --out out.png --width W --height H "
                 "[--sharpness S]\n");
    return 1;
  }

  fsr1::Image src = LoadPng(in_path);
  fsr1::Image upscaled = fsr1::easu_upscale(src, out_width, out_height);
  fsr1::Image result = have_sharpness ? fsr1::rcas_sharpen(upscaled, sharpness) : upscaled;
  SavePng(out_path, result);
  return 0;
}
