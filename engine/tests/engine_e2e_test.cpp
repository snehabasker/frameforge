// Engine output vs Python reference on a fixed sequence, tolerance-based
// (CLAUDE.md 7.4). Generates a fresh fixed-seed model + PyTorch reference
// output via scripts/export_reference_sequence.py, then runs the SAME
// ONNX export through the C++ engine's actual pipeline
// (frameforge::RunSequence: EXR read -> tonemap -> depth-normalize ->
// recurrent ONNX inference) and compares.
//
// Skips itself (not a failure) if no Python interpreter with the
// `frameforge` package installed can be found -- this keeps the C++ CI
// jobs Python-free (CLAUDE.md 8's CI constraints are scoped around the
// `python` job for that), while still giving real, strong local
// verification (as used throughout this repo's own development -- see
// docs/DECISIONS.md).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "frameforge/ort_session.hpp"
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

bool RunPythonReferenceExport(const std::string& python, const std::string& seq_parent_dir,
                               const std::string& out_dir) {
#ifdef FRAMEFORGE_PROJECT_ROOT
  std::string root = FRAMEFORGE_PROJECT_ROOT;
  std::string inner = "\"" + python + "\" \"" + root +
                       "/scripts/export_reference_sequence.py\" --seq-dir \"" + seq_parent_dir +
                       "\" --out \"" + out_dir + "\" --config \"" + root + "/configs/data.yaml\"";
  // Windows cmd.exe quirk: when a command passed to `system()` starts with
  // a quoted, space-containing path, cmd.exe only parses it correctly if
  // the WHOLE command line is wrapped in one extra pair of quotes (it
  // otherwise treats the first quoted token as literal text, not a
  // program path -- easy to hit any time a repo lives somewhere like
  // "C:/Users/Jane Doe/..." as this one's dev environment does).
#ifdef _WIN32
  std::string cmd = "\"" + inner + "\"";
#else
  std::string cmd = inner;
#endif
  int rc = std::system(cmd.c_str());
  return rc == 0;
#else
  (void)python; (void)seq_parent_dir; (void)out_dir;
  return false;
#endif
}

}  // namespace

TEST(EngineE2E, MatchesPythonReferenceOnFixedSequence) {
  std::string python = FindPython();
  if (python.empty()) {
    GTEST_SKIP() << "no project .venv Python found; skipping cross-language reference test "
                    "(this is a local-verification test, not required in C++ CI -- see "
                    "this test file's header comment)";
  }

  fs::path out_dir = fs::temp_directory_path() / "frameforge_engine_e2e_test";
  fs::create_directories(out_dir);
  std::string seq_parent = std::string(FRAMEFORGE_FIXTURES_DIR) + "/sequences";

  bool ok = RunPythonReferenceExport(python, seq_parent, out_dir.string());
  if (!ok) {
    GTEST_SKIP() << "export_reference_sequence.py failed to run (see stderr above); "
                     "skipping rather than failing on an environment issue unrelated to "
                     "the engine code itself";
  }

  std::ifstream meta(out_dir / "meta.txt");
  ASSERT_TRUE(meta.good());
  int t, h, w;
  meta >> t >> h >> w;
  ASSERT_GT(t, 0);

  std::ifstream ref_file(out_dir / "reference_output.bin", std::ios::binary);
  ASSERT_TRUE(ref_file.good());
  std::vector<float> reference(static_cast<size_t>(t) * 3 * h * w);
  ref_file.read(reinterpret_cast<char*>(reference.data()),
                static_cast<std::streamsize>(reference.size() * sizeof(float)));
  ASSERT_TRUE(ref_file.good());

  frameforge::ModelIoSpec spec;
  frameforge::OrtModelRunner runner((out_dir / "model.onnx").string(), spec,
                                     frameforge::ExecutionProvider::kCPU, 1);
  auto outputs = frameforge::RunSequence(seq_parent + "/synthetic_pan_00/lr", runner);

  ASSERT_EQ(static_cast<int>(outputs.size()), t);

  float max_abs_error = 0.0f;
  size_t offset = 0;
  for (int frame_idx = 0; frame_idx < t; ++frame_idx) {
    const auto& out = outputs[frame_idx];
    ASSERT_EQ(out.h(), h);
    ASSERT_EQ(out.w(), w);
    for (size_t i = 0; i < out.size(); ++i) {
      float diff = std::abs(out.data()[i] - reference[offset + i]);
      max_abs_error = std::max(max_abs_error, diff);
    }
    offset += out.size();
  }

  // Looser than the pure-ONNX-parity tolerance (1e-4, test_onnx_parity.py)
  // since this also exercises the engine's own from-scratch tonemap/
  // depth-normalize/bilinear-upsample C++ implementations against
  // PyTorch's, which accumulate a little more floating-point drift.
  EXPECT_LT(max_abs_error, 5e-3f) << "engine output diverged from Python reference";
}
