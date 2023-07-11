
#include "band/device/cpu.h"

#include <gtest/gtest.h>

#include "band/device/util.h"

namespace band {
namespace test {
struct CPUMaskFixture : public testing::TestWithParam<CPUMaskFlag> {};

// NOTE: set may be different from kAll due to device-specific limitation
// e.g., Galaxy S20 can only set affinity to first 6 cores
#if BAND_SUPPORT_DEVICE
TEST_P(CPUMaskFixture, AffinitySetTest) {
  CpuSet target_set = BandCPUMaskGetSet(GetParam());
  // this fails if target_set is null
  absl::Status set_status = SetCPUThreadAffinity(target_set);

  CpuSet current_set;
  // should always success
  EXPECT_EQ(GetCPUThreadAffinity(current_set), absl::OkStatus());
  if (!set_status.ok()) {
    EXPECT_EQ(target_set.NumEnabled(), 0);
  }
}

TEST(CPUTest, DisableTest) {
  CpuSet set = BandCPUMaskGetSet(CPUMaskFlag::kAll);
  EXPECT_EQ(SetCPUThreadAffinity(set), absl::OkStatus());

  for (size_t i = 0; i < GetCPUCount(); i++) {
    set.Disable(i);
  }

  EXPECT_TRUE(!SetCPUThreadAffinity(set).ok());
}

TEST(CPUTest, EnableTest) {
  CpuSet set;
  EXPECT_TRUE(!SetCPUThreadAffinity(set).ok());

  for (size_t i = 0; i < GetCPUCount(); i++) {
    set.Enable(i);
  }

  EXPECT_EQ(SetCPUThreadAffinity(set), absl::OkStatus());
  EXPECT_EQ(GetCPUThreadAffinity(set), absl::OkStatus());
}

TEST_P(CPUMaskFixture, FrequencyCPUSetTest) {
  CpuSet target_set = BandCPUMaskGetSet(GetParam());
  if (target_set.NumEnabled() == 0) {
    EXPECT_EQ(cpu::GetTargetFrequencyKhz(target_set).value(), 0);
    EXPECT_EQ(cpu::GetTargetMinFrequencyKhz(target_set).value(), 0);
    std::vector<size_t> available_frequencies =
        cpu::GetAvailableFrequenciesKhz(target_set).value();
    EXPECT_EQ(available_frequencies.size(), 0);
    if (device::IsRooted()) {
      EXPECT_EQ(cpu::GetTargetMaxFrequencyKhz(target_set).value(), 0);
      EXPECT_EQ(cpu::GetFrequencyKhz(target_set).value(), 0);
    }
    EXPECT_EQ(cpu::GetUpTransitionLatencyMs(target_set).value(), 0);
    EXPECT_EQ(cpu::GetDownTransitionLatencyMs(target_set).value(), 0);
    EXPECT_EQ(cpu::GetTotalTransitionCount(target_set).value(), 0);
  } else {
    EXPECT_TRUE(cpu::GetTargetFrequencyKhz(target_set).ok());
    EXPECT_TRUE(cpu::GetTargetMinFrequencyKhz(target_set).ok());
    EXPECT_TRUE(cpu::GetAvailableFrequenciesKhz(target_set).ok());
    if (device::IsRooted()) {
      EXPECT_TRUE(cpu::GetTargetMaxFrequencyKhz(target_set).ok());
      EXPECT_TRUE(cpu::GetFrequencyKhz(target_set).ok());
    }
    EXPECT_TRUE(cpu::GetUpTransitionLatencyMs(target_set).ok());
    EXPECT_TRUE(cpu::GetDownTransitionLatencyMs(target_set).ok());
    EXPECT_TRUE(cpu::GetTotalTransitionCount(target_set).ok());
  }
}
#endif

TEST_P(CPUMaskFixture, FrequencyCPUSetStatusTest) {
  CpuSet target_set = BandCPUMaskGetSet(GetParam());
  absl::Status expected_status =
      device::SupportsDevice() ? absl::OkStatus()
                               : absl::UnavailableError("Device not supported");

  EXPECT_EQ(SetCPUThreadAffinity(target_set), expected_status);
  EXPECT_EQ(GetCPUThreadAffinity(target_set), expected_status);
  if (device::IsRooted()) {
    EXPECT_EQ(cpu::GetTargetFrequencyKhz(target_set).status(), expected_status);
    EXPECT_EQ(cpu::GetTargetMaxFrequencyKhz(target_set).status(),
              expected_status);
  }
  EXPECT_EQ(cpu::GetTargetMinFrequencyKhz(target_set).status(),
            expected_status);
  EXPECT_EQ(cpu::GetAvailableFrequenciesKhz(target_set).status(),
            expected_status);
  EXPECT_EQ(cpu::GetUpTransitionLatencyMs(target_set).status(),
            expected_status);
  EXPECT_EQ(cpu::GetDownTransitionLatencyMs(target_set).status(),
            expected_status);
  EXPECT_EQ(cpu::GetTotalTransitionCount(target_set).status(), expected_status);
}

INSTANTIATE_TEST_SUITE_P(MaskSetTests, CPUMaskFixture,
                         testing::Values(CPUMaskFlag::kAll,
                                         CPUMaskFlag::kLittle,
                                         CPUMaskFlag::kBig,
                                         CPUMaskFlag::kPrimary));

}  // namespace test
}  // namespace band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
