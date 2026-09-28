// Cross-checked against python/frameforge/colorspace.py's
// tonemap_and_encode() -- values below were computed by an actual run of
// that Python function (see the command in this test's comments), not
// hand-derived, per CLAUDE.md 3.3's "use the same transform everywhere."
#include <gtest/gtest.h>

#include "frameforge/tonemap.hpp"

namespace {
constexpr float kTol = 1e-5f;
}

// python -c "import torch; from frameforge.colorspace import tonemap_and_encode;
//   print(tonemap_and_encode(torch.tensor([X])).item())"  for each X below.
TEST(Tonemap, MatchesPythonReferenceValues) {
  EXPECT_NEAR(frameforge::TonemapAndEncode(0.0f), 0.00000000f, kTol);
  EXPECT_NEAR(frameforge::TonemapAndEncode(0.5f), 0.61250103f, kTol);
  EXPECT_NEAR(frameforge::TonemapAndEncode(1.0f), 0.73535693f, kTol);
  EXPECT_NEAR(frameforge::TonemapAndEncode(2.0f), 0.83600694f, kTol);
  EXPECT_NEAR(frameforge::TonemapAndEncode(10.0f), 0.95892423f, kTol);
  EXPECT_NEAR(frameforge::TonemapAndEncode(100.0f), 0.99563497f, kTol);
}

TEST(Tonemap, SrgbRoundTrip) {
  for (float x = 0.0f; x <= 1.0f; x += 0.05f) {
    float roundtrip = frameforge::SrgbToLinear(frameforge::LinearToSrgb(x));
    EXPECT_NEAR(roundtrip, x, 1e-4f);
  }
}

TEST(Tonemap, SrgbEndpoints) {
  EXPECT_NEAR(frameforge::LinearToSrgb(0.0f), 0.0f, 1e-6f);
  EXPECT_NEAR(frameforge::LinearToSrgb(1.0f), 1.0f, 1e-6f);
}

TEST(Tonemap, ReinhardBoundedAndMonotonic) {
  float prev = -1.0f;
  for (float x = 0.0f; x <= 1000.0f; x += 10.0f) {
    float v = frameforge::ReinhardTonemap(x);
    EXPECT_GE(v, 0.0f);
    EXPECT_LT(v, 1.0f);
    EXPECT_GT(v, prev);
    prev = v;
  }
}
