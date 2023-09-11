// Copyright 2023 Seoul National University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "band/device/cpu.h"

#include <gtest/gtest.h>

#include "band/common.h"
#include "band/device/util.h"
#include "band/logger.h"

namespace band {
namespace test {

#if BAND_IS_MOBILE
// NOTE: set may be different from kAll due to device-specific limitation
// e.g., Galaxy S20 can only set affinity to first 6 cores
TEST(CPUTest, AffinitySetTest) {
  std::vector<CPUMaskFlag> masks = {CPUMaskFlag::kAll, CPUMaskFlag::kLittle,
                                    CPUMaskFlag::kBig, CPUMaskFlag::kPrimary};
  for (auto mask : masks) {
    CpuSet target_set = BandCPUMaskGetSet(mask);
    // this fails if target_set is null
    absl::Status set_status = SetCPUThreadAffinity(target_set);
    if (!set_status.ok()) {
      EXPECT_EQ(target_set.NumEnabled(), 0);
    } else {
      sleep(1);
      CpuSet current_set;
      // should always success
      EXPECT_EQ(GetCPUThreadAffinity(current_set), absl::OkStatus());
      EXPECT_EQ(target_set, current_set);
    }
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
#endif
}  // namespace test

}  // namespace band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
