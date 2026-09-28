// Tiled vs untiled equivalence (CLAUDE.md 7.3: "a test that the tiled
// output matches the untiled output within tolerance"). Uses the same
// export_reference_sequence.py pattern as engine_e2e_test.cpp/pipeline_test.cpp
// to get a real (temporal, so lr_depth/lr_mv/prev_hr_output tiling all get
// exercised) ONNX model; skips if no project .venv Python is found, same
// reasoning as those tests.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "frameforge/exr_io.hpp"
#include "frameforge/ort_session.hpp"
#include "frameforge/preprocess.hpp"
#include "frameforge/tile.hpp"

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

TEST(Tile, TiledMatchesUntiledWithinTolerance) {
  std::string python = FindPython();
  if (python.empty()) {
    GTEST_SKIP() << "no project .venv Python found; see engine_e2e_test.cpp's header comment";
  }

  fs::path out_dir = fs::temp_directory_path() / "frameforge_tile_test";
  fs::create_directories(out_dir);
  std::string seq_parent = std::string(FRAMEFORGE_FIXTURES_DIR) + "/sequences";
  if (!ExportModel(python, seq_parent, out_dir.string())) {
    GTEST_SKIP() << "export_reference_sequence.py failed to run";
  }

  frameforge::ModelIoSpec spec;
  frameforge::OrtModelRunner runner((out_dir / "model.onnx").string(), spec,
                                     frameforge::ExecutionProvider::kCPU, 1);

  frameforge::ChannelMap channel_map;
  auto frame1 = frameforge::LoadFramePasses(seq_parent + "/synthetic_pan_00/lr/frame_0001.exr",
                                             channel_map);
  frameforge::Tensor lr_rgb = frameforge::ApplyTonemap(frame1.combined);
  std::vector<frameforge::Tensor> depths = {frame1.depth};
  frameforge::NormalizeDepthInPlace(depths);
  frameforge::Tensor lr_mv = frame1.vector_to_previous;
  frameforge::Tensor prev_hr = frameforge::BilinearUpsample2x(lr_rgb);

  frameforge::Tensor untiled = runner.Run(lr_rgb, &depths[0], &lr_mv, &prev_hr);
  frameforge::Tensor tiled = frameforge::RunTiled(runner, lr_rgb, &depths[0], &lr_mv, &prev_hr,
                                                   /*tile_size=*/24, /*overlap=*/8);

  ASSERT_EQ(tiled.h(), untiled.h());
  ASSERT_EQ(tiled.w(), untiled.w());

  float max_abs_error = 0.0f;
  double sum_abs_error = 0.0;
  for (size_t i = 0; i < untiled.size(); ++i) {
    float diff = std::abs(tiled.data()[i] - untiled.data()[i]);
    max_abs_error = std::max(max_abs_error, diff);
    sum_abs_error += diff;
  }
  double mean_abs_error = sum_abs_error / static_cast<double>(untiled.size());

  // Feathered tile blending is expected to differ slightly from the
  // untiled path near tile seams (different receptive field content at
  // the tile boundary); mean error should still be small.
  EXPECT_LT(mean_abs_error, 0.05) << "tiled/untiled mean abs error too high";
  EXPECT_LT(max_abs_error, 0.5f) << "tiled/untiled max abs error too high";
}

TEST(Tile, ZeroTileSizeRunsUntiledIdentically) {
  std::string python = FindPython();
  if (python.empty()) {
    GTEST_SKIP() << "no project .venv Python found";
  }
  fs::path out_dir = fs::temp_directory_path() / "frameforge_tile_test_zero";
  fs::create_directories(out_dir);
  std::string seq_parent = std::string(FRAMEFORGE_FIXTURES_DIR) + "/sequences";
  if (!ExportModel(python, seq_parent, out_dir.string())) {
    GTEST_SKIP() << "export_reference_sequence.py failed to run";
  }

  frameforge::ModelIoSpec spec;
  frameforge::OrtModelRunner runner((out_dir / "model.onnx").string(), spec,
                                     frameforge::ExecutionProvider::kCPU, 1);
  frameforge::ChannelMap channel_map;
  auto frame1 = frameforge::LoadFramePasses(seq_parent + "/synthetic_pan_00/lr/frame_0001.exr",
                                             channel_map);
  frameforge::Tensor lr_rgb = frameforge::ApplyTonemap(frame1.combined);
  std::vector<frameforge::Tensor> depths = {frame1.depth};
  frameforge::NormalizeDepthInPlace(depths);
  frameforge::Tensor lr_mv = frame1.vector_to_previous;
  frameforge::Tensor prev_hr = frameforge::BilinearUpsample2x(lr_rgb);

  frameforge::Tensor a = runner.Run(lr_rgb, &depths[0], &lr_mv, &prev_hr);
  frameforge::Tensor b = frameforge::RunTiled(runner, lr_rgb, &depths[0], &lr_mv, &prev_hr,
                                               /*tile_size=*/0, /*overlap=*/8);
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_FLOAT_EQ(a.data()[i], b.data()[i]);
  }
}
