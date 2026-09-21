// Golden-file test for the FSR1 CPU port (CLAUDE.md 7.4: "FSR 1 port:
// deterministic output on a fixed image (golden file)").
//
// A small, fixed, purely synthetic input pattern (no file I/O, so this
// test has zero external dependencies) is upscaled and sharpened; the
// result must (a) be bit-identical across repeated runs (determinism) and
// (b) match a handful of spot-checked pixel values captured from a real
// run of this same code (not hand-derived), so a future change to the
// algorithm that shifts the output gets caught.
#include <cmath>

#include <gtest/gtest.h>

#include "fsr1/fsr1.hpp"

namespace {

fsr1::Image MakeFixedTestImage() {
  // 8x8 gradient + checkerboard mix -- has both smooth regions (exercises
  // EASU's low-gradient path) and hard edges (exercises the edge/direction
  // detection path), which is the point of a golden test for this filter.
  fsr1::Image img(8, 8);
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      float gradient = static_cast<float>(x + y) / 14.0f;
      float checker = ((x / 2 + y / 2) % 2 == 0) ? 1.0f : 0.0f;
      float v = 0.5f * gradient + 0.5f * checker;
      img.at(x, y, 0) = v;
      img.at(x, y, 1) = v * 0.8f;
      img.at(x, y, 2) = v * 0.6f;
    }
  }
  return img;
}

constexpr float kTol = 1e-5f;

}  // namespace

TEST(Fsr1Golden, EasuOutputIsDeterministic) {
  fsr1::Image src = MakeFixedTestImage();
  fsr1::Image out1 = fsr1::easu_upscale(src, 16, 16);
  fsr1::Image out2 = fsr1::easu_upscale(src, 16, 16);
  ASSERT_EQ(out1.data.size(), out2.data.size());
  for (size_t i = 0; i < out1.data.size(); ++i) {
    EXPECT_EQ(out1.data[i], out2.data[i]) << "mismatch at index " << i;
  }
}

TEST(Fsr1Golden, EasuOutputShapeAndRange) {
  fsr1::Image src = MakeFixedTestImage();
  fsr1::Image out = fsr1::easu_upscale(src, 16, 16);
  EXPECT_EQ(out.width, 16);
  EXPECT_EQ(out.height, 16);
  for (float v : out.data) {
    EXPECT_GE(v, -0.05f);  // EASU can slightly overshoot at sharp edges
    EXPECT_LE(v, 1.05f);
    EXPECT_TRUE(std::isfinite(v));
  }
}

TEST(Fsr1Golden, EasuMatchesCapturedReferenceValues) {
  // Values captured from an actual build-and-run of this exact code (not
  // hand computed): configure+build engine/, run fsr1_tests, and if this
  // test ever legitimately needs updating (an intentional algorithm
  // change), regenerate by reading the failure diff's "evaluates to" value.
  fsr1::Image src = MakeFixedTestImage();
  fsr1::Image out = fsr1::easu_upscale(src, 16, 16);

  struct Spot { int x, y, c; float expected; };
  const Spot spots[] = {
      {0, 0, 0, 0.500000f},
      {8, 8, 0, 0.374184f},
      {15, 15, 0, 1.000000f},
      {4, 4, 1, 0.185061f},
      {12, 3, 2, 0.359619f},
  };
  for (const auto& s : spots) {
    EXPECT_NEAR(out.at(s.x, s.y, s.c), s.expected, kTol)
        << "at (" << s.x << "," << s.y << "," << s.c << ")";
  }
}

TEST(Fsr1Golden, RcasOutputIsDeterministicAndBounded) {
  fsr1::Image src = MakeFixedTestImage();
  fsr1::Image upscaled = fsr1::easu_upscale(src, 16, 16);
  fsr1::Image sharp1 = fsr1::rcas_sharpen(upscaled, 0.0f);
  fsr1::Image sharp2 = fsr1::rcas_sharpen(upscaled, 0.0f);
  for (size_t i = 0; i < sharp1.data.size(); ++i) {
    EXPECT_EQ(sharp1.data[i], sharp2.data[i]);
  }
  for (float v : sharp1.data) {
    EXPECT_TRUE(std::isfinite(v));
  }
}

TEST(Fsr1Golden, RcasZeroSharpnessSharpensMoreThanHighStopReduction) {
  // Higher `sharpness` argument = more stops of REDUCTION -> less sharp.
  // A flat-ish region shouldn't move much either way, but a strongly-edged
  // region should show more deviation from the unsharpened image at
  // sharpness=0 than at sharpness=4.
  fsr1::Image src = MakeFixedTestImage();
  fsr1::Image upscaled = fsr1::easu_upscale(src, 16, 16);
  fsr1::Image sharp_max = fsr1::rcas_sharpen(upscaled, 0.0f);
  fsr1::Image sharp_min = fsr1::rcas_sharpen(upscaled, 4.0f);

  double diff_max = 0.0, diff_min = 0.0;
  for (size_t i = 0; i < upscaled.data.size(); ++i) {
    diff_max += std::abs(sharp_max.data[i] - upscaled.data[i]);
    diff_min += std::abs(sharp_min.data[i] - upscaled.data[i]);
  }
  EXPECT_GT(diff_max, diff_min);
}
