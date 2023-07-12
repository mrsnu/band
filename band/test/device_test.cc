
#include <gtest/gtest.h>

#include "band/device/generic.h"
#include "band/device/util.h"

namespace band {
namespace test {

struct DeviceFlagFixture : public testing::TestWithParam<DeviceFlag> {};

TEST_P(DeviceFlagFixture, FrequencyStatusTest) {
  DeviceFlag flag = GetParam();
  auto is_ok = [](absl::Status status) -> bool {
    if (device::SupportsDevice()) {
      return status.ok() || status.code() == absl::StatusCode::kNotFound;
    } else {
      return status.code() == absl::StatusCode::kUnavailable;
    }
  };

  EXPECT_TRUE(is_ok(generic::GetMinFrequencyKhz(flag).status()));
  EXPECT_TRUE(is_ok(generic::GetMaxFrequencyKhz(flag).status()));
  EXPECT_TRUE(is_ok(generic::GetFrequencyKhz(flag).status()));
  EXPECT_TRUE(is_ok(generic::GetPollingIntervalMs(flag).status()));
  EXPECT_TRUE(is_ok(generic::GetTargetFrequencyKhz(flag).status()));
  EXPECT_TRUE(is_ok(generic::GetAvailableFrequenciesKhz(flag).status()));
  EXPECT_TRUE(is_ok(generic::GetClockStats(flag).status()));
}

INSTANTIATE_TEST_SUITE_P(AllDeviceFlag, DeviceFlagFixture,
                         testing::Values(DeviceFlag::kCPU, DeviceFlag::kGPU,
                                         DeviceFlag::kDSP, DeviceFlag::kNPU));

}  // namespace test
}  // namespace band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
