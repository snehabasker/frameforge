#include "frameforge/sequence_runner.hpp"

#include <algorithm>
#include <filesystem>

#include "frameforge/preprocess.hpp"

namespace fs = std::filesystem;

namespace frameforge {

namespace {

std::vector<fs::path> ListFrames(const fs::path& dir) {
  std::vector<fs::path> frames;
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (entry.path().extension() == ".exr") {
      frames.push_back(entry.path());
    }
  }
  std::sort(frames.begin(), frames.end());
  return frames;
}

}  // namespace

std::vector<Tensor> RunSequence(const std::string& seq_dir, OrtModelRunner& runner,
                                 const ChannelMap& channel_map) {
  auto frame_paths = ListFrames(seq_dir);
  if (frame_paths.empty()) {
    throw std::runtime_error("no frame_*.exr files found in " + seq_dir);
  }

  std::vector<RawFramePasses> raw_frames;
  raw_frames.reserve(frame_paths.size());
  for (const auto& p : frame_paths) {
    raw_frames.push_back(LoadFramePasses(p.string(), channel_map));
  }

  std::vector<Tensor> depths;
  for (auto& f : raw_frames) depths.push_back(f.depth);
  NormalizeDepthInPlace(depths);

  std::vector<Tensor> outputs;
  outputs.reserve(raw_frames.size());

  Tensor prev_hr_output;
  bool have_prev = false;

  for (size_t t = 0; t < raw_frames.size(); ++t) {
    Tensor lr_rgb = ApplyTonemap(raw_frames[t].combined);
    Tensor* lr_depth_ptr = nullptr;
    Tensor* lr_mv_ptr = nullptr;
    Tensor* prev_ptr = nullptr;
    Tensor lr_mv;

    if (runner.spec().use_depth) {
      lr_depth_ptr = &depths[t];
    }
    if (runner.spec().use_temporal) {
      lr_mv = raw_frames[t].vector_to_previous;
      if (t == 0) {
        std::fill(lr_mv.data(), lr_mv.data() + lr_mv.size(), 0.0f);
      }
      lr_mv_ptr = &lr_mv;

      // The warp happens INSIDE the ONNX graph (see student.py / export_onnx.py);
      // pass the raw previous output, never pre-warp it here.
      if (!have_prev) {
        prev_hr_output = BilinearUpsample2x(lr_rgb);
      }
      prev_ptr = &prev_hr_output;
    }

    Tensor output = runner.Run(lr_rgb, lr_depth_ptr, lr_mv_ptr, prev_ptr);
    prev_hr_output = output;
    have_prev = true;
    outputs.push_back(output);
  }
  return outputs;
}

}  // namespace frameforge
