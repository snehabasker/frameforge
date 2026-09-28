// Shared "run a whole LR sequence through the model" logic, used by both
// the CLI (src/main.cpp) and the engine-vs-Python-reference test
// (tests/engine_e2e_test.cpp), so there's exactly one implementation of
// the tonemap -> depth-normalize -> recurrent-inference pipeline.
#pragma once

#include <string>
#include <vector>

#include "frameforge/exr_io.hpp"
#include "frameforge/ort_session.hpp"
#include "frameforge/tensor.hpp"

namespace frameforge {

// Reads every frame_*.exr in `seq_dir` (sorted), tonemaps, per-sequence
// depth-normalizes, and runs each frame through `runner` in order,
// maintaining the recurrent prev_hr_output state exactly as
// python/frameforge/train.run_unroll does. Returns one HR output tensor
// per frame, in order.
std::vector<Tensor> RunSequence(const std::string& seq_dir, OrtModelRunner& runner,
                                 const ChannelMap& channel_map = ChannelMap{});

}  // namespace frameforge
