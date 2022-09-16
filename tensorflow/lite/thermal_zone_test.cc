#include "tensorflow/lite/thermal_zone.h"

#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace {

// All the tests in this file is for Pixel 4 XL device.
// In order to test for other devices, proper sysfs paths should be given.

TEST(ThermalZoneManagerTest, CheckPathSanityTest) {
  impl::ThermalZoneManager& manager = impl::ThermalZoneManager::instance();
  EXPECT_EQ(manager.CheckPathSanity("/sys/class/thermal/tz-by-name/cpu-0-0-usr/temp"), true);
  EXPECT_EQ(manager.CheckPathSanity("/foobarbaz"), false);
}

TEST(ThermalZoneManagerTest, SetCPUPathTest) {
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

TEST(ThermalZoneManagerTest, GetCPUTemperatureTest) {
  impl::ThermalZoneManager& manager = impl::ThermalZoneManager::instance();
  impl::thermal_t temp = manager.GetTemperature("CPU0");
  TFLITE_LOG_PROD(TFLITE_LOG_INFO, "CPU0's temp: %d", temp);
  EXPECT_GE(temp, 10000);

  temp = manager.GetTemperature("CPU1");
  TFLITE_LOG_PROD(TFLITE_LOG_INFO, "CPU1's temp: %d", temp);
  EXPECT_GE(temp, 10000);

  temp = manager.GetTemperature("CPU2");
  TFLITE_LOG_PROD(TFLITE_LOG_INFO, "CPU2's temp: %d", temp);
  EXPECT_GE(temp, 10000);

  temp = manager.GetTemperature("CPU3");
  TFLITE_LOG_PROD(TFLITE_LOG_INFO, "CPU3's temp: %d", temp);
  EXPECT_GE(temp, 10000);

  temp = manager.GetTemperature("CPU4");
  TFLITE_LOG_PROD(TFLITE_LOG_INFO, "CPU4's temp: %d", temp);
  EXPECT_GE(temp, 10000);

  temp = manager.GetTemperature("CPU5");
  TFLITE_LOG_PROD(TFLITE_LOG_INFO, "CPU5's temp: %d", temp);
  EXPECT_GE(temp, 10000);

  temp = manager.GetTemperature("CPU6");
  TFLITE_LOG_PROD(TFLITE_LOG_INFO, "CPU6's temp: %d", temp);
  EXPECT_GE(temp, 10000);

  temp = manager.GetTemperature("CPU7");
  TFLITE_LOG_PROD(TFLITE_LOG_INFO, "CPU7's temp: %d", temp);
  EXPECT_GE(temp, 10000);
}

TEST(ThermalZoneManagerTest, SetGPUPathTest) {

}

TEST(ThermalZoneManagerTest, SetNPUPathTest) {

}

TEST(ThermalZoneManagerTest, SetDSPPathTest) {

}

TEST(ThermalZoneManagerTest, SetWifiPathTest) {

}

TEST(ThermalZoneManagerTest, SetCellularPathTest) {

}



} // namespace
} // namespace tflite

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}