
#include "band/cpu.h"

#include <gtest/gtest.h>

namespace band {
namespace test {
struct AffinityMasksFixture : public testing::TestWithParam<BandCPUMaskFlags> {
};

#ifdef _BAND_SUPPORT_THREAD_AFFINITY
TEST_P(AffinityMasksFixture, AffinitySetTest) {
  CpuSet target_set = BandCPUMaskGetSet(GetParam());
  // this fails if target_set is null
  BandStatus set_status = SetCPUThreadAffinity(target_set);

  CpuSet current_set;
  // should always success
  EXPECT_EQ(GetCPUThreadAffinity(current_set), kBandOk);
  if (set_status == kBandOk) {
    EXPECT_EQ(target_set, current_set);
    EXPECT_EQ(target_set.NumEnabled(), current_set.NumEnabled());
  } else {
    EXPECT_EQ(target_set.NumEnabled(), 0);
  }
}

TEST(CPUTest, DisableTest) {
  CpuSet set = BandCPUMaskGetSet(kBandAll);
  EXPECT_EQ(SetCPUThreadAffinity(set), kBandOk);

  for (size_t i = 0; i < GetCPUCount(); i++) {
    set.Disable(i);
  }

  EXPECT_EQ(SetCPUThreadAffinity(set), kBandError);
}

TEST(CPUTest, EnableTest) {
  CpuSet set;
  EXPECT_EQ(SetCPUThreadAffinity(set), kBandError);

  for (size_t i = 0; i < GetCPUCount(); i++) {
    set.Enable(i);
  }

  EXPECT_EQ(SetCPUThreadAffinity(set), kBandOk);
  EXPECT_EQ(GetCPUThreadAffinity(set), kBandOk);
  EXPECT_EQ(set, BandCPUMaskGetSet(kBandAll));
}

INSTANTIATE_TEST_SUITE_P(AffinitySetTests, AffinityMasksFixture,
                         testing::Values(kBandAll, kBandLittle, kBandBig,
                                         kBandPrimary));
#else

TEST_P(AffinityMasksFixture, DummyTest) {
  CpuSet target_set = BandCPUMaskGetSet(GetParam());
  // always success, as this platform not supports thread affinity
  EXPECT_EQ(SetCPUThreadAffinity(target_set), kBandOk);
  EXPECT_EQ(GetCPUThreadAffinity(target_set), kBandOk);
}

INSTANTIATE_TEST_SUITE_P(DummyTest, AffinityMasksFixture,
                         testing::Values(kBandAll, kBandLittle, kBandBig,
                                         kBandPrimary));
#endif

}  // namespace test
}  // namespace band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
