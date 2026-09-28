#include "frameforge/exr_io.hpp"

#define TINYEXR_IMPLEMENTATION
#include "tinyexr/tinyexr.h"

namespace frameforge {

namespace {

// RAII wrapper so a thrown exception mid-load can't leak tinyexr's
// manually-managed header/image structs (CLAUDE.md 7.3: RAII throughout).
class ExrHandles {
 public:
  ExrHandles() {
    InitEXRHeader(&header_);
    InitEXRImage(&image_);
  }
  ~ExrHandles() {
    FreeEXRImage(&image_);
    FreeEXRHeader(&header_);
  }
  ExrHandles(const ExrHandles&) = delete;
  ExrHandles& operator=(const ExrHandles&) = delete;

  EXRHeader header_;
  EXRImage image_;
};

}  // namespace

ExrFrame ReadExrFrame(const std::string& path) {
  EXRVersion version;
  if (ParseEXRVersionFromFile(&version, path.c_str()) != TINYEXR_SUCCESS) {
    throw std::runtime_error("failed to parse EXR version: " + path);
  }

  ExrHandles handles;
  const char* err = nullptr;
  if (ParseEXRHeaderFromFile(&handles.header_, &version, path.c_str(), &err) != TINYEXR_SUCCESS) {
    std::string msg = err ? err : "unknown error";
    if (err) FreeEXRErrorMessage(err);
    throw std::runtime_error("failed to parse EXR header (" + path + "): " + msg);
  }

  // Force everything to float32 on load, regardless of stored precision
  // (half or float) -- matches the Python loader's `dtype=np.float32`.
  for (int i = 0; i < handles.header_.num_channels; ++i) {
    if (handles.header_.pixel_types[i] == TINYEXR_PIXELTYPE_HALF) {
      handles.header_.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
    }
  }

  if (LoadEXRImageFromFile(&handles.image_, &handles.header_, path.c_str(), &err) !=
      TINYEXR_SUCCESS) {
    std::string msg = err ? err : "unknown error";
    if (err) FreeEXRErrorMessage(err);
    throw std::runtime_error("failed to load EXR image (" + path + "): " + msg);
  }

  ExrFrame frame;
  frame.width = handles.image_.width;
  frame.height = handles.image_.height;
  const size_t n_pixels = static_cast<size_t>(frame.width) * frame.height;

  for (int c = 0; c < handles.image_.num_channels; ++c) {
    const std::string name = handles.header_.channels[c].name;
    const float* src = reinterpret_cast<const float*>(handles.image_.images[c]);
    frame.channels[name] = std::vector<float>(src, src + n_pixels);
  }
  return frame;
}

namespace {

std::vector<float> RequireChannel(const ExrFrame& frame, const std::string& name) {
  return frame.Channel(name);
}

}  // namespace

RawFramePasses LoadFramePasses(const std::string& path, const ChannelMap& map) {
  ExrFrame frame = ReadExrFrame(path);
  const int h = frame.height;
  const int w = frame.width;

  RawFramePasses out;
  out.combined = Tensor(1, 3, h, w);
  out.depth = Tensor(1, 1, h, w);
  out.vector_to_previous = Tensor(1, 2, h, w);

  const std::string vl = map.view_layer;
  auto r = RequireChannel(frame, vl + "." + map.combined_pass + ".R");
  auto g = RequireChannel(frame, vl + "." + map.combined_pass + ".G");
  auto b = RequireChannel(frame, vl + "." + map.combined_pass + ".B");
  auto z = RequireChannel(frame, vl + "." + map.depth_pass + ".Z");
  auto vx = RequireChannel(frame, vl + "." + map.vector_pass + "." + map.vector_to_previous_x);
  auto vy = RequireChannel(frame, vl + "." + map.vector_pass + "." + map.vector_to_previous_y);

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      size_t idx = static_cast<size_t>(y) * w + x;
      out.combined.at(0, 0, y, x) = r[idx];
      out.combined.at(0, 1, y, x) = g[idx];
      out.combined.at(0, 2, y, x) = b[idx];
      out.depth.at(0, 0, y, x) = z[idx];
      out.vector_to_previous.at(0, 0, y, x) = vx[idx];
      out.vector_to_previous.at(0, 1, y, x) = vy[idx];
    }
  }
  return out;
}

}  // namespace frameforge
