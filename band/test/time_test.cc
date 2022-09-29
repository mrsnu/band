
#include "band/time.h"
#include <gtest/gtest.h>

namespace Band {
namespace Time {
namespace Test {

TEST(TimeTest, NowMicros) {
  auto now0 = NowMicros();
  EXPECT_GT(now0, 0);
  auto now1 = NowMicros();
  EXPECT_GE(now1, now0);
}

TEST(TimeTest, SleepForMicros) {
  // A zero sleep shouldn't cause issues.
  SleepForMicros(0);

  // Sleeping should be reflected in the current time.
  auto now0 = NowMicros();
  SleepForMicros(50);
  auto now1 = NowMicros();
  EXPECT_GE(now1, now0 + 50);

  // Sleeping more than a second should function properly.
  now0 = NowMicros();
  SleepForMicros(1e6 + 50);
  now1 = NowMicros();
  EXPECT_GE(now1, now0 + 1e6 + 50);
}

} // namespace Test
} // namespace Time
} // namespace Band

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
