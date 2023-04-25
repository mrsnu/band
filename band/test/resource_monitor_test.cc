#include "band/resource_monitor.h"

#include <gtest/gtest.h>

namespace band {
namespace test {

TEST(ResourceMonitorTest, CreationTest) {
  auto status_or_monitor = ResourceMonitor::Create();
  EXPECT_EQ(status_or_monitor.status(), absl::OkStatus());
  auto monitor = status_or_monitor.value();

  auto tzs = monitor.GetDetectedThermalZonePaths();
  auto cpufreqs = monitor.GetDetectedCpuFreqPaths();
  auto devfreqs = monitor.GetDetectedDevFreqPaths();

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
#ifdef __ANDROID__
  std::cout << "Detected devfreqs: " << devfreqs.size() << std::endl;
  for (auto& devfreq : devfreqs) {
    std::cout << devfreq << std::endl;
  }
  EXPECT_GT(devfreqs.size(), 0);
#endif  // __ANDROID__
}

TEST(ResourceMonitorTest, ThermalZoneTest) {
  auto status_or_monitor = ResourceMonitor::Create();
  EXPECT_EQ(status_or_monitor.status(), absl::OkStatus());
  auto monitor = status_or_monitor.value();

  auto status_or_thermal_status = monitor.GetCurrentThermalStatus();
  EXPECT_EQ(status_or_thermal_status.status(), absl::OkStatus());
  auto thermal_status = status_or_thermal_status.value();
  for (auto& tz : thermal_status) {
    std::cout << tz.first << ": " << tz.second << std::endl;
  }
}

// May require sudo privilege.
TEST(ResourceMonitorTest, FreqTest) {
  auto status_or_monitor = ResourceMonitor::Create();
  EXPECT_EQ(status_or_monitor.status(), absl::OkStatus());
  auto monitor = status_or_monitor.value();

  auto status_or_freqs = monitor.GetCurrentFrequency();
  EXPECT_EQ(status_or_freqs.status(), absl::OkStatus());
  auto freqs = status_or_freqs.value();
}

}  // namespace test
}  // namespace band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}