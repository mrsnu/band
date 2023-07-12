#include "band/resource_monitor.h"

#include <gtest/gtest.h>

namespace band {
namespace test {

TEST(ResourceMonitorTest, CreationTest) {
  auto& monitor = ResourceMonitor::Create();

  auto tzs = monitor.GetDetectedThermalZonePaths();
  auto cpufreqs = monitor.GetDetectedCpuFreqPaths();
  auto devfreqs = monitor.GetDetectedDevFreqPaths();

#ifdef __ANDROID__
  std::cout << "Detected thermal zones: " << tzs.size() << std::endl;
  for (auto& tz : tzs) {
    std::cout << tz << std::endl;
  }
  EXPECT_GT(tzs.size(), 0);
  std::cout << "Detected cpufreqs: " << cpufreqs.size() << std::endl;
  for (auto& cpufreq : cpufreqs) {
    std::cout << cpufreq << std::endl;
  }
  EXPECT_GT(cpufreqs.size(), 0);
  std::cout << "Detected devfreqs: " << devfreqs.size() << std::endl;
  for (auto& devfreq : devfreqs) {
    std::cout << devfreq << std::endl;
  }
  EXPECT_GT(devfreqs.size(), 0);
#endif  // __ANDROID__
}

TEST(ResourceMonitorTest, ThermalZoneTest) {
  auto& monitor = ResourceMonitor::Create();

  auto status_or_thermal = monitor.GetCurrentThermal();
  EXPECT_EQ(status_or_thermal.status(), absl::OkStatus());
  auto thermal = status_or_thermal.value();
  for (auto& tz : thermal.status) {
    std::cout << tz.first << ": " << tz.second << std::endl;
  }
}

// May require sudo privilege.
TEST(ResourceMonitorTest, FreqTest) {
  auto& monitor = ResourceMonitor::Create();

  auto status_or_freqs = monitor.GetCurrentFrequency();
#ifdef __ANDROID__
  EXPECT_EQ(status_or_freqs.status(), absl::OkStatus());
  auto freqs = status_or_freqs.value();
  for (auto& freq : freqs.status) {
    std::cout << freq.first << ": " << freq.second << std::endl;
  }
#endif  // __ANDROID__
}

}  // namespace test
}  // namespace band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}