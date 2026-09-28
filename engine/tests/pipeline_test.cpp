// Threaded pipeline correctness: RunSequencePipelined must produce
// byte-identical output to the sequential frameforge::RunSequence (same
// math, only the execution overlaps across reader/inference/writer
// threads) -- this is what actually proves the SpscRingBuffer hookup and
// thread handoff didn't change behavior, on top of queue_test.cpp's
// lower-level queue correctness checks.
//
// Reuses scripts/export_reference_sequence.py just to get a small
// deterministic ONNX model on disk (no C++ ONNX-authoring API is used
// anywhere in this project); skips itself if no project .venv Python is
// found, same as engine_e2e_test.cpp, for the same CI-scoping reason (see
// that file's header comment).
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "frameforge/ort_session.hpp"
#include "frameforge/pipeline.hpp"
#include "frameforge/png_io.hpp"
#include "frameforge/sequence_runner.hpp"

namespace fs = std::filesystem;

namespace {

std::string FindPython() {
#ifdef FRAMEFORGE_PROJECT_ROOT
  fs::path venv_py = fs::path(FRAMEFORGE_PROJECT_ROOT) / ".venv" / "Scripts" / "python.exe";
  if (fs::exists(venv_py)) return venv_py.string();
  venv_py = fs::path(FRAMEFORGE_PROJECT_ROOT) / ".venv" / "bin" / "python";
  if (fs::exists(venv_py)) return venv_py.string();
#endif
  return "";
}

bool ExportModel(const std::string& python, const std::string& seq_parent_dir,
                  const std::string& out_dir) {
#ifdef FRAMEFORGE_PROJECT_ROOT
  std::string root = FRAMEFORGE_PROJECT_ROOT;
  std::string inner = "\"" + python + "\" \"" + root +
                       "/scripts/export_reference_sequence.py\" --seq-dir \"" + seq_parent_dir +
                       "\" --out \"" + out_dir + "\" --config \"" + root + "/configs/data.yaml\"";
#ifdef _WIN32
  std::string cmd = "\"" + inner + "\"";
#else
  std::string cmd = inner;
#endif
  return std::system(cmd.c_str()) == 0;
#else
  (void)python; (void)seq_parent_dir; (void)out_dir;
  return false;
#endif
}

}  // namespace

TEST(Pipeline, PipelinedMatchesSequentialExactly) {
  std::string python = FindPython();
  if (python.empty()) {
    GTEST_SKIP() << "no project .venv Python found; see engine_e2e_test.cpp's header comment";
  }

  fs::path out_dir = fs::temp_directory_path() / "frameforge_pipeline_test";
  fs::create_directories(out_dir);
  std::string seq_parent = std::string(FRAMEFORGE_FIXTURES_DIR) + "/sequences";

  if (!ExportModel(python, seq_parent, out_dir.string())) {
    GTEST_SKIP() << "export_reference_sequence.py failed to run";
  }

  std::string model_path = (out_dir / "model.onnx").string();
  std::string seq_dir = seq_parent + "/synthetic_pan_00/lr";

  frameforge::ModelIoSpec spec;
  frameforge::OrtModelRunner sequential_runner(model_path, spec, frameforge::ExecutionProvider::kCPU, 1);
  auto sequential_outputs = frameforge::RunSequence(seq_dir, sequential_runner);

  frameforge::OrtModelRunner pipelined_runner(model_path, spec, frameforge::ExecutionProvider::kCPU, 1);
  fs::path pipelined_out = out_dir / "pipelined_pngs";
  frameforge::RunSequencePipelined(seq_dir, pipelined_out.string(), pipelined_runner);

  for (size_t i = 0; i < sequential_outputs.size(); ++i) {
    char name_buf[64];
    std::snprintf(name_buf, sizeof(name_buf), "frame_%04zu.png", i);
    frameforge::Tensor from_pipeline = frameforge::ReadPngAsFloat((pipelined_out / name_buf).string());

    // Compare through the same PNG quantization the sequential path would
    // see if also written to PNG, so this is an apples-to-apples check
    // (both 8-bit-quantized), not float-vs-float.
    fs::path seq_png = out_dir / "sequential_reference.png";
    frameforge::WritePng(seq_png.string(), sequential_outputs[i]);
    frameforge::Tensor from_sequential = frameforge::ReadPngAsFloat(seq_png.string());

    ASSERT_EQ(from_pipeline.h(), from_sequential.h());
    ASSERT_EQ(from_pipeline.w(), from_sequential.w());
    for (size_t p = 0; p < from_pipeline.size(); ++p) {
      ASSERT_FLOAT_EQ(from_pipeline.data()[p], from_sequential.data()[p])
          << "frame " << i << " pixel " << p << " diverged between pipelined and sequential";
    }
  }
}
