
#include "band/device/cpu.h"

#include <gtest/gtest.h>

namespace band {
namespace test {
struct AffinityMasksFixture : public testing::TestWithParam<CPUMaskFlag> {};

#ifdef _BAND_SUPPORT_THREAD_AFFINITY
TEST_P(AffinityMasksFixture, AffinitySetTest) {
  CpuSet target_set = BandCPUMaskGetSet(GetParam());
  // this fails if target_set is null
  absl::Status set_status = SetCPUThreadAffinity(target_set);

  CpuSet current_set;
  // should always success
  EXPECT_EQ(GetCPUThreadAffinity(current_set), absl::OkStatus());
  if (set_status.ok()) {
    EXPECT_EQ(target_set, current_set);
    EXPECT_EQ(target_set.NumEnabled(), current_set.NumEnabled());
  } else {
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
  EXPECT_EQ(set, BandCPUMaskGetSet(CPUMaskFlag::kAll));
}

INSTANTIATE_TEST_SUITE_P(AffinitySetTests, AffinityMasksFixture,
                         testing::Values(CPUMaskFlag::kAll,
                                         CPUMaskFlag::kLittle,
                                         CPUMaskFlag::kBig,
                                         CPUMaskFlag::kPrimary));
#else

TEST_P(AffinityMasksFixture, DummyTest) {
  CpuSet target_set = BandCPUMaskGetSet(GetParam());
  // always success, as this platform not supports thread affinity
  EXPECT_EQ(SetCPUThreadAffinity(target_set), absl::OkStatus());
  EXPECT_EQ(GetCPUThreadAffinity(target_set), absl::OkStatus());
}

INSTANTIATE_TEST_SUITE_P(DummyTest, AffinityMasksFixture,
                         testing::Values(CPUMaskFlag::kAll,
                                         CPUMaskFlag::kLittle,
                                         CPUMaskFlag::kBig,
                                         CPUMaskFlag::kPrimary));
#endif

}  // namespace test
}  // namespace band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
