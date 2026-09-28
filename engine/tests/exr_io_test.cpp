// Cross-checked against python/frameforge/data/exr_io.py reading the SAME
// committed fixture file -- values captured from an actual run of
// `load_frame_passes(...)` on frame_0001.exr (see the comment above each
// assertion for the exact Python call), not hand-derived.
#include <string>

#include <gtest/gtest.h>

#include "frameforge/exr_io.hpp"

namespace {
std::string FixturePath(const std::string& rel) {
  return std::string(FRAMEFORGE_FIXTURES_DIR) + "/" + rel;
}
constexpr float kTol = 1e-4f;
}  // namespace

TEST(ExrIo, ReadFrameShapeMatchesPython) {
  auto frame = frameforge::ReadExrFrame(
      FixturePath("sequences/synthetic_pan_00/lr/frame_0001.exr"));
  EXPECT_EQ(frame.width, 64);
  EXPECT_EQ(frame.height, 36);
}

// python: load_frame_passes(".../frame_0001.exr", cfg["exr_channel_map"])["combined"][0,0]
// -> [0.39092368 0.8108883  0.9675001 ]
TEST(ExrIo, LoadFramePassesCombinedMatchesPython) {
  frameforge::ChannelMap map;
  auto passes = frameforge::LoadFramePasses(
      FixturePath("sequences/synthetic_pan_00/lr/frame_0001.exr"), map);

  EXPECT_NEAR(passes.combined.at(0, 0, 0, 0), 0.39092368f, kTol);
  EXPECT_NEAR(passes.combined.at(0, 1, 0, 0), 0.8108883f, kTol);
  EXPECT_NEAR(passes.combined.at(0, 2, 0, 0), 0.9675001f, kTol);

  // python: combined[10,20] (row=y=10, col=x=20)
  EXPECT_NEAR(passes.combined.at(0, 0, 10, 20), 0.49971256f, kTol);
  EXPECT_NEAR(passes.combined.at(0, 1, 10, 20), 0.5242886f, kTol);
  EXPECT_NEAR(passes.combined.at(0, 2, 10, 20), 0.14421129f, kTol);
}

// python: depth[5,5] -> 2.5692356
TEST(ExrIo, LoadFramePassesDepthMatchesPython) {
  frameforge::ChannelMap map;
  auto passes = frameforge::LoadFramePasses(
      FixturePath("sequences/synthetic_pan_00/lr/frame_0001.exr"), map);
  EXPECT_NEAR(passes.depth.at(0, 0, 5, 5), 2.5692356f, kTol);
}

// python: vector_to_previous[0,0] -> [-2. 1.], [10,10] -> [-2. 1.]
// (this fixture has a constant synthetic pan, so the vector is uniform)
TEST(ExrIo, LoadFramePassesVectorMatchesPython) {
  frameforge::ChannelMap map;
  auto passes = frameforge::LoadFramePasses(
      FixturePath("sequences/synthetic_pan_00/lr/frame_0001.exr"), map);
  EXPECT_NEAR(passes.vector_to_previous.at(0, 0, 0, 0), -2.0f, kTol);
  EXPECT_NEAR(passes.vector_to_previous.at(0, 1, 0, 0), 1.0f, kTol);
  EXPECT_NEAR(passes.vector_to_previous.at(0, 0, 10, 10), -2.0f, kTol);
  EXPECT_NEAR(passes.vector_to_previous.at(0, 1, 10, 10), 1.0f, kTol);
}

TEST(ExrIo, MissingChannelThrows) {
  frameforge::ChannelMap map;
  map.combined_pass = "NoSuchPass";
  EXPECT_THROW(
      frameforge::LoadFramePasses(
          FixturePath("sequences/synthetic_pan_00/lr/frame_0001.exr"), map),
      std::exception);
}

TEST(ExrIo, MissingFileThrows) {
  EXPECT_THROW(frameforge::ReadExrFrame(FixturePath("no_such_file.exr")), std::exception);
}
