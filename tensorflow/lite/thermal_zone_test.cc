#include "tensorflow/lite/thermal_zone.h"

#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace {

// All the tests in this file is for Pixel 4 XL device.
// In order to test for other devices, proper sysfs paths should be given.

TEST(ThermalZoneManagerTest, SetPathTest) {
  impl::ThermalZoneManager& manager = impl::ThermalZoneManager::instance();
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

TEST(ThermalZoneManagerTest, GetPathTest) {
  impl::ThermalZoneManager& manager = impl::ThermalZoneManager::instance();
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

TEST(ThermalZoneManagerTest, GetCPUTemperatureTest) {
  impl::ThermalZoneManager& manager = impl::ThermalZoneManager::instance();
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

TEST(ThermalZoneManagerTest, GetTemperatureHistoryAllTest) {
  impl::ThermalZoneManager& manager = impl::ThermalZoneManager::instance();
  manager.GetTemperature("CPU0");
  manager.GetTemperature("CPU0");
  manager.GetTemperature("CPU0");
  manager.GetTemperature("CPU0");

  std::vector<impl::thermal_t> temp_history = manager.GetTemperatureHistory("CPU0");
  EXPECT_EQ(manager.GetTemperatureHistory("CPU0", 0), temp_history[0]);
  EXPECT_EQ(manager.GetTemperatureHistory("CPU0", 1), temp_history[1]);
  EXPECT_EQ(manager.GetTemperatureHistory("CPU0", 2), temp_history[2]);
  EXPECT_EQ(manager.GetTemperatureHistory("CPU0", 3), temp_history[3]);
}

TEST(ThermalZoneManagerTest, ClearHistoryTest) {
  impl::ThermalZoneManager& manager = impl::ThermalZoneManager::instance();
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