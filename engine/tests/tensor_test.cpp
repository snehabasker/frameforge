// Tensor layout conversion tests (CLAUDE.md 7.4: "Tensor layout conversion
// (HWC <-> NCHW, tonemap round-trip)").
#include <gtest/gtest.h>

#include "frameforge/tensor.hpp"

TEST(Tensor, BasicIndexing) {
  frameforge::Tensor t(1, 3, 4, 5);
  t.at(0, 1, 2, 3) = 42.0f;
  EXPECT_FLOAT_EQ(t.at(0, 1, 2, 3), 42.0f);
  EXPECT_EQ(t.size(), 1u * 3 * 4 * 5);
}

TEST(Tensor, HwcToNchwRoundTrip) {
  const int h = 3, w = 4, c = 3;
  std::vector<float> hwc(h * w * c);
  for (size_t i = 0; i < hwc.size(); ++i) hwc[i] = static_cast<float>(i);

  frameforge::Tensor nchw = frameforge::HwcToNchw(hwc, h, w, c);
  EXPECT_EQ(nchw.n(), 1);
  EXPECT_EQ(nchw.c(), c);
  EXPECT_EQ(nchw.h(), h);
  EXPECT_EQ(nchw.w(), w);

  std::vector<float> back = frameforge::NchwToHwc(nchw);
  ASSERT_EQ(back.size(), hwc.size());
  for (size_t i = 0; i < hwc.size(); ++i) {
    EXPECT_FLOAT_EQ(back[i], hwc[i]) << "mismatch at flat index " << i;
  }
}

TEST(Tensor, HwcToNchwKnownValue) {
  // 2x2 image, 2 channels: pixel (y=1,x=0) channel 1 should be a known value.
  int h = 2, w = 2, c = 2;
  std::vector<float> hwc = {
      // (y=0,x=0): ch0,ch1    (y=0,x=1): ch0,ch1
      1.0f, 2.0f, 3.0f, 4.0f,
      // (y=1,x=0): ch0,ch1    (y=1,x=1): ch0,ch1
      5.0f, 6.0f, 7.0f, 8.0f,
  };
  frameforge::Tensor nchw = frameforge::HwcToNchw(hwc, h, w, c);
  EXPECT_FLOAT_EQ(nchw.at(0, 0, 0, 0), 1.0f);
  EXPECT_FLOAT_EQ(nchw.at(0, 1, 0, 0), 2.0f);
  EXPECT_FLOAT_EQ(nchw.at(0, 0, 1, 0), 5.0f);
  EXPECT_FLOAT_EQ(nchw.at(0, 1, 1, 1), 8.0f);
}
