// M0 sanity test: proves GoogleTest is wired up via FetchContent and the
// engine's own headers are reachable from the tests target.
#include <gtest/gtest.h>

#include "frameforge/version.hpp"

TEST(Skeleton, VersionIsPositive) {
  EXPECT_GE(frameforge::kVersionMajor, 0);
  EXPECT_GE(frameforge::kVersionMinor, 0);
  EXPECT_GT(frameforge::kVersionPatch, 0);
}

TEST(Skeleton, MilestoneTagIsSet) {
  EXPECT_STREQ(frameforge::kMilestone, "M0-skeleton");
}
