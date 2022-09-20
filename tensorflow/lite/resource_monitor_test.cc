#include "tensorflow/lite/resource_monitor.h"

#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace {

// All the tests in this file is for Pixel 4 XL device.
// In order to test for other devices, proper sysfs paths should be given.

TEST(ResourceMonitorTest, SetPathTest) {
  impl::ResourceMonitor& manager = impl::ResourceMonitor::instance();
  TfLiteStatus status = manager.SetThermalZonePath(
      "CPU0", 
      "/sys/class/thermal/tz-by-name/cpu-1-0-usr/temp");
  EXPECT_EQ(status, kTfLiteOk);
  status = manager.SetThermalZonePath(
      "CPU1", 
      "/sys/class/thermal/tz-by-name/cpu-1-1-usr/temp");
  EXPECT_EQ(status, kTfLiteOk);
  status = manager.SetThermalZonePath(
      "CPU2", 
      "/sys/class/thermal/tz-by-name/cpu-1-2-usr/temp");
  EXPECT_EQ(status, kTfLiteOk);
  status = manager.SetThermalZonePath(
      "CPU3", 
      "/sys/class/thermal/tz-by-name/cpu-1-3-usr/temp");
  EXPECT_EQ(status, kTfLiteOk);
  status = manager.SetThermalZonePath(
      "CPU4", 
      "/sys/class/thermal/tz-by-name/cpu-1-4-usr/temp");
  EXPECT_EQ(status, kTfLiteOk);
  status = manager.SetThermalZonePath(
      "CPU5", 
      "/sys/class/thermal/tz-by-name/cpu-1-5-usr/temp");
  EXPECT_EQ(status, kTfLiteOk);
  status = manager.SetThermalZonePath(
      "CPU6", 
      "/sys/class/thermal/tz-by-name/cpu-1-6-usr/temp");
  EXPECT_EQ(status, kTfLiteOk);
  status = manager.SetThermalZonePath(
      "CPU7", 
      "/sys/class/thermal/tz-by-name/cpu-1-7-usr/temp");
  EXPECT_EQ(status, kTfLiteOk);
}

TEST(ResourceMonitorTest, GetPathTest) {
  impl::ResourceMonitor& manager = impl::ResourceMonitor::instance();
  EXPECT_EQ(manager.GetThermalZonePath("CPU0"),
            "/sys/class/thermal/tz-by-name/cpu-1-0-usr/temp");

  EXPECT_EQ(manager.GetThermalZonePath("CPU1"),
            "/sys/class/thermal/tz-by-name/cpu-1-1-usr/temp");

  EXPECT_EQ(manager.GetThermalZonePath("CPU2"),
            "/sys/class/thermal/tz-by-name/cpu-1-2-usr/temp");

  EXPECT_EQ(manager.GetThermalZonePath("CPU3"),
            "/sys/class/thermal/tz-by-name/cpu-1-3-usr/temp");

  EXPECT_EQ(manager.GetThermalZonePath("CPU4"),
            "/sys/class/thermal/tz-by-name/cpu-1-4-usr/temp");

  EXPECT_EQ(manager.GetThermalZonePath("CPU5"),
            "/sys/class/thermal/tz-by-name/cpu-1-5-usr/temp");

  EXPECT_EQ(manager.GetThermalZonePath("CPU6"),
            "/sys/class/thermal/tz-by-name/cpu-1-6-usr/temp");

  EXPECT_EQ(manager.GetThermalZonePath("CPU7"),
            "/sys/class/thermal/tz-by-name/cpu-1-7-usr/temp");
}

TEST(ResourceMonitorTest, GetCPUTemperatureTest) {
  impl::ResourceMonitor& manager = impl::ResourceMonitor::instance();
  impl::thermal_t temp = manager.GetTemperature("CPU0");
  EXPECT_GE(temp, 10000);

  temp = manager.GetTemperature("CPU1");
  EXPECT_GE(temp, 10000);

  temp = manager.GetTemperature("CPU2");
  EXPECT_GE(temp, 10000);

  temp = manager.GetTemperature("CPU3");
  EXPECT_GE(temp, 10000);

  temp = manager.GetTemperature("CPU4");
  EXPECT_GE(temp, 10000);

  temp = manager.GetTemperature("CPU5");
  EXPECT_GE(temp, 10000);

  temp = manager.GetTemperature("CPU6");
  EXPECT_GE(temp, 10000);

  temp = manager.GetTemperature("CPU7");
  EXPECT_GE(temp, 10000);
}

TEST(ResourceMonitorTest, GetTemperatureHistoryAllTest) {
  impl::ResourceMonitor& manager = impl::ResourceMonitor::instance();
  manager.GetTemperature("CPU0");
  manager.GetTemperature("CPU0");
  manager.GetTemperature("CPU0");
  manager.GetTemperature("CPU0");

  std::vector<impl::ThermalInfo> temp_history = manager.GetTemperatureHistory("CPU0");
  EXPECT_EQ(manager.GetTemperatureHistory("CPU0", 0).temperature, temp_history[0].temperature);
  EXPECT_EQ(manager.GetTemperatureHistory("CPU0", 1).temperature, temp_history[1].temperature);
  EXPECT_EQ(manager.GetTemperatureHistory("CPU0", 2).temperature, temp_history[2].temperature);
  EXPECT_EQ(manager.GetTemperatureHistory("CPU0", 3).temperature, temp_history[3].temperature);
}

TEST(ResourceMonitorTest, ClearHistoryTest) {
  impl::ResourceMonitor& manager = impl::ResourceMonitor::instance();
  manager.ClearHistory("CPU0");
  EXPECT_TRUE(manager.GetTemperatureHistory("CPU0").empty());
  manager.ClearHistoryAll();
  EXPECT_TRUE(manager.GetTemperatureHistory("CPU4").empty());
}

} // namespace
} // namespace tflite

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}