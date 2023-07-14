#include "band/resource_monitor.h"

#include <gtest/gtest.h>

namespace band {
namespace test {

TEST(ResourceMonitorTest, CreationTest) {
  ResourceMonitorConfig config{"", {}, 10};
  ResourceMonitor monitor;
  auto status = monitor.Init(config);

#if BAND_IS_MOBILE
  EXPECT_TRUE(status.ok());
  size_t num_tz = monitor.NumThermalResources(ThermalFlag::TZ_TEMPERATURE);
  std::cout << "Found " << num_tz << " thermal zones" << std::endl;

  auto tz_paths = monitor.GetThermalPaths();
  std::cout << "Thermal zone paths: " << std::endl;
  for (auto& path : tz_paths) {
    std::cout << path << std::endl;
  }
  auto devfreq_paths = monitor.GetDevFreqPaths();
  std::cout << "Devfreq paths: " << std::endl;
  for (auto& path : devfreq_paths) {
    std::cout << path << std::endl;
  }
  auto cpu_freq_paths = monitor.GetCpuFreqPaths();
  std::cout << "CPU freq paths: " << std::endl;
  for (auto& path : cpu_freq_paths) {
    std::cout << path << std::endl;
  }
#endif  // __ANDROID__
}

}  // namespace test
}  // namespace band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}