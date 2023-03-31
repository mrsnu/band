
#include "band/cpu.h"

#include <gtest/gtest.h>

namespace Band {
namespace Test {
struct AffinityMasksFixture : public testing::TestWithParam<CPUMaskFlags> {
};

#ifdef _BAND_SUPPORT_THREAD_AFFINITY
TEST_P(AffinityMasksFixture, AffinitySetTest) {
  CpuSet target_set = BandCPUMaskGetSet(GetParam());
  // this fails if target_set is null
  absl::Status set_status = SetCPUThreadAffinity(target_set);

  CpuSet current_set;
  // should always success
  EXPECT_TRUE(GetCPUThreadAffinity(current_set).ok());
  if (set_status.ok()) {
    EXPECT_EQ(target_set, current_set);
    EXPECT_EQ(target_set.NumEnabled(), current_set.NumEnabled());
  } else {
    EXPECT_EQ(target_set.NumEnabled(), 0);
  }
}

TEST(CPUTest, DisableTest) {
  CpuSet set = BandCPUMaskGetSet(CPUMaskFlags::All);
  EXPECT_TRUE(SetCPUThreadAffinity(set).ok());

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

  EXPECT_TRUE(SetCPUThreadAffinity(set).ok());
  EXPECT_TRUE(GetCPUThreadAffinity(set).ok());
  EXPECT_EQ(set, BandCPUMaskGetSet(CPUMaskFlags::All));
}

INSTANTIATE_TEST_SUITE_P(AffinitySetTests, AffinityMasksFixture,
                         testing::Values(CPUMaskFlags::All, CPUMaskFlags::Little, CPUMaskFlags::Big,
                                         CPUMaskFlags::Primary));
#else

TEST_P(AffinityMasksFixture, DummyTest) {
  CpuSet target_set = BandCPUMaskGetSet(GetParam());
  // always success, as this platform not supports thread affinity
  EXPECT_TRUE(SetCPUThreadAffinity(target_set).ok());
  EXPECT_TRUE(GetCPUThreadAffinity(target_set).ok());
}

INSTANTIATE_TEST_SUITE_P(DummyTest, AffinityMasksFixture,
                         testing::Values(CPUMaskFlags::All, CPUMaskFlags::Little, CPUMaskFlags::Big,
                                         CPUMaskFlags::Primary));
#endif

}  // namespace Test
}  // namespace Band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
