#include "band/device/frequency.h"

#include "band/test/device/device_util.h"

#include <gtest/gtest.h>

namespace band {
namespace test {

TEST(BandTestDeviceFrequencyTest, FreqTest) {
  auto config = GetPixel4DeviceConfig();
  Frequency frequency(config);
  auto avail_freqs = frequency.GetAllAvailableFrequency();
  for (int i = 0; i < EnumLength<DeviceFlag>(); i++) {
    auto device_flag = static_cast<DeviceFlag>(i);
    for (const auto& freq : avail_freqs[device_flag]) {
      EXPECT_EQ(frequency.SetFrequency(device_flag, freq), absl::OkStatus());
      EXPECT_EQ(frequency.GetFrequency(device_flag), freq);
    }
  }

  for (const auto& freq : frequency.GetRuntimeAvailableFrequency()) {
    EXPECT_EQ(frequency.SetRuntimeFrequency(freq), absl::OkStatus());
    EXPECT_EQ(frequency.GetRuntimeFrequency(), freq);
  }
}

}  // namespace test
}  // namespace band

int main(int argc, char** argv) {
#ifdef __ANDROID__
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
#endif  // __ANDROID__
  return 0;
}