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

#include "band/resource_monitor.h"

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

namespace band {
namespace test {

TEST(ResourceMonitorTest, CreationTest) {
  ResourceMonitorConfig config{"", {}, 10};
  ResourceMonitor monitor;
  EXPECT_EQ(monitor.Init(config), absl::OkStatus());

#if BAND_IS_MOBILE
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

  bool callback_called = false;
  monitor.AddOnUpdate(
      [&callback_called](const ResourceMonitor&) { callback_called = true; });
  std::this_thread::sleep_for(std::chrono::milliseconds(22));
  EXPECT_TRUE(callback_called);
#endif  // BAND_IS_MOBILE
}

TEST(ResourceMonitorTest, GetThermalTest) {
  ResourceMonitorConfig config{"", {}, 10};
  ResourceMonitor monitor;
  EXPECT_EQ(monitor.Init(config), absl::OkStatus());

#if BAND_IS_MOBILE
  size_t num_tz = monitor.NumThermalResources(ThermalFlag::TZ_TEMPERATURE);
  for (size_t i = 0; i < num_tz; ++i) {
    EXPECT_EQ(monitor.AddThermalResource(ThermalFlag::TZ_TEMPERATURE, i),
              absl::OkStatus());
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(22));

  for (size_t i = 0; i < num_tz; ++i) {
    auto temp = monitor.GetThermal(ThermalFlag::TZ_TEMPERATURE, i);
    EXPECT_TRUE(temp.ok());
    std::cout << "Thermal " << i << ": " << temp.value() << std::endl;
  }
#endif  // BAND_IS_MOBILE
}

TEST(ResourceMonitorTest, GetDevFreqTest) {
  ResourceMonitorConfig config{"", {}, 10};
  ResourceMonitor monitor;
  EXPECT_EQ(monitor.Init(config), absl::OkStatus());

#if BAND_IS_MOBILE
  std::vector<DeviceFlag> valid_devices;
  for (size_t i = 0; i < EnumLength<DeviceFlag>(); ++i) {
    if (monitor.IsValidDevice(static_cast<DeviceFlag>(i))) {
      valid_devices.push_back(static_cast<DeviceFlag>(i));

      EXPECT_EQ(monitor.AddDevFreqResource(static_cast<DeviceFlag>(i),
                                           DevFreqFlag::CUR_FREQ),
                absl::OkStatus());
      EXPECT_EQ(monitor.AddDevFreqResource(static_cast<DeviceFlag>(i),
                                           DevFreqFlag::TARGET_FREQ),
                absl::OkStatus());
      EXPECT_EQ(monitor.AddDevFreqResource(static_cast<DeviceFlag>(i),
                                           DevFreqFlag::MIN_FREQ),
                absl::OkStatus());
      EXPECT_EQ(monitor.AddDevFreqResource(static_cast<DeviceFlag>(i),
                                           DevFreqFlag::MAX_FREQ),
                absl::OkStatus());
      EXPECT_EQ(monitor.AddDevFreqResource(static_cast<DeviceFlag>(i),
                                           DevFreqFlag::POLLING_INTERVAL),
                absl::OkStatus());
    }
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(22));

  for (auto& valid_device : valid_devices) {
    auto cur_freq = monitor.GetDevFreq(valid_device, DevFreqFlag::CUR_FREQ);
    EXPECT_TRUE(cur_freq.ok());
    std::cout << "DevFreq " << ToString(valid_device)
              << " CUR_FREQ: " << cur_freq.value() << std::endl;
    auto target_freq =
        monitor.GetDevFreq(valid_device, DevFreqFlag::TARGET_FREQ);
    EXPECT_TRUE(target_freq.ok());
    std::cout << "DevFreq " << ToString(valid_device)
              << " TARGET_FREQ: " << target_freq.value() << std::endl;
    auto min_freq = monitor.GetDevFreq(valid_device, DevFreqFlag::MIN_FREQ);
    EXPECT_TRUE(min_freq.ok());
    std::cout << "DevFreq " << ToString(valid_device)
              << " MIN_FREQ: " << min_freq.value() << std::endl;
    auto max_freq = monitor.GetDevFreq(valid_device, DevFreqFlag::MAX_FREQ);
    EXPECT_TRUE(max_freq.ok());
    std::cout << "DevFreq " << ToString(valid_device)
              << " MAX_FREQ: " << max_freq.value() << std::endl;
    auto polling_interval =
        monitor.GetDevFreq(valid_device, DevFreqFlag::POLLING_INTERVAL);
    EXPECT_TRUE(polling_interval.ok());
    std::cout << "DevFreq " << ToString(valid_device)
              << " POLLING_INTERVAL: " << polling_interval.value() << std::endl;
  }
#endif  // BAND_IS_MOBILE
}

TEST(ResourceMonitorTest, GetCpuFreqTest) {
  ResourceMonitorConfig config{"", {}, 10};
  ResourceMonitor monitor;
  EXPECT_EQ(monitor.Init(config), absl::OkStatus());

#if BAND_IS_MOBILE
  std::vector<CPUMaskFlag> valid_cpus;
  for (size_t i = 0; i < EnumLength<CPUMaskFlag>(); i++) {
    const CPUMaskFlag flag = static_cast<CPUMaskFlag>(i);
    if (flag == CPUMaskFlag::kAll) {
      continue;
    }

    valid_cpus.push_back(flag);
    EXPECT_EQ(monitor.AddCpuFreqResource(flag, CpuFreqFlag::CUR_FREQ),
              absl::OkStatus());
    EXPECT_EQ(monitor.AddCpuFreqResource(flag, CpuFreqFlag::TARGET_FREQ),
              absl::OkStatus());
    EXPECT_EQ(monitor.AddCpuFreqResource(flag, CpuFreqFlag::MIN_FREQ),
              absl::OkStatus());
    EXPECT_EQ(monitor.AddCpuFreqResource(flag, CpuFreqFlag::MAX_FREQ),
              absl::OkStatus());
    EXPECT_EQ(
        monitor.AddCpuFreqResource(flag, CpuFreqFlag::UP_TRANSITION_LATENCY),
        absl::OkStatus());
    EXPECT_EQ(
        monitor.AddCpuFreqResource(flag, CpuFreqFlag::DOWN_TRANSITION_LATENCY),
        absl::OkStatus());
    // This is optional. E.g., Pixel 4 doesn't have this.
    auto status =
        monitor.AddCpuFreqResource(flag, CpuFreqFlag::TRANSITION_COUNT);
    std::cout << "AddCpuFreqResource " << ToString(flag)
              << " TRANSITION_COUNT: " << status << std::endl;
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(22));

  for (auto& cpu_mask : valid_cpus) {
    auto cur_freq = monitor.GetCpuFreq(cpu_mask, CpuFreqFlag::CUR_FREQ);
    EXPECT_TRUE(cur_freq.ok());
    std::cout << "CpuFreq " << ToString(cpu_mask)
              << " CUR_FREQ: " << cur_freq.value() << std::endl;
    auto target_freq = monitor.GetCpuFreq(cpu_mask, CpuFreqFlag::TARGET_FREQ);
    EXPECT_TRUE(target_freq.ok());
    std::cout << "CpuFreq " << ToString(cpu_mask)
              << " TARGET_FREQ: " << target_freq.value() << std::endl;
    auto min_freq = monitor.GetCpuFreq(cpu_mask, CpuFreqFlag::MIN_FREQ);
    EXPECT_TRUE(min_freq.ok());
    std::cout << "CpuFreq " << ToString(cpu_mask)
              << " MIN_FREQ: " << min_freq.value() << std::endl;
    auto max_freq = monitor.GetCpuFreq(cpu_mask, CpuFreqFlag::MAX_FREQ);
    EXPECT_TRUE(max_freq.ok());
    std::cout << "CpuFreq " << ToString(cpu_mask)
              << " MAX_FREQ: " << max_freq.value() << std::endl;
    auto up_transition_latency =
        monitor.GetCpuFreq(cpu_mask, CpuFreqFlag::UP_TRANSITION_LATENCY);
    EXPECT_TRUE(up_transition_latency.ok());
    std::cout << "CpuFreq " << ToString(cpu_mask)
              << " UP_TRANSITION_LATENCY: " << up_transition_latency.value()
              << std::endl;
    auto down_transition_latency =
        monitor.GetCpuFreq(cpu_mask, CpuFreqFlag::DOWN_TRANSITION_LATENCY);
    EXPECT_TRUE(down_transition_latency.ok());
    std::cout << "CpuFreq " << ToString(cpu_mask)
              << " DOWN_TRANSITION_LATENCY: " << down_transition_latency.value()
              << std::endl;
    auto transition_count =
        monitor.GetCpuFreq(cpu_mask, CpuFreqFlag::TRANSITION_COUNT);
    // This is optional, so we don't check if it's ok.
    if (transition_count.ok()) {
      std::cout << "CpuFreq " << ToString(cpu_mask)
                << " TRANSITION_COUNT: " << transition_count.value()
                << std::endl;
    }
  }
#endif
}

TEST(ResourceMonitorTest, LogTest) {
  std::string log_path = "/data/local/tmp/example_log.json";
#if BAND_IS_MOBILE
  {
    ResourceMonitorConfig config{log_path, {}, 10};
    ResourceMonitor monitor;
    EXPECT_EQ(monitor.Init(config), absl::OkStatus());
    // add some resources
    size_t num_tz = monitor.NumThermalResources(ThermalFlag::TZ_TEMPERATURE);
    for (size_t i = 0; i < num_tz; ++i) {
      EXPECT_EQ(monitor.AddThermalResource(ThermalFlag::TZ_TEMPERATURE, i),
                absl::OkStatus());
    }

    std::vector<DeviceFlag> valid_devices;
    for (size_t i = 0; i < EnumLength<DeviceFlag>(); ++i) {
      if (monitor.IsValidDevice(static_cast<DeviceFlag>(i))) {
        valid_devices.push_back(static_cast<DeviceFlag>(i));

        EXPECT_EQ(monitor.AddDevFreqResource(static_cast<DeviceFlag>(i),
                                             DevFreqFlag::CUR_FREQ),
                  absl::OkStatus());
        EXPECT_EQ(monitor.AddDevFreqResource(static_cast<DeviceFlag>(i),
                                             DevFreqFlag::TARGET_FREQ),
                  absl::OkStatus());
        EXPECT_EQ(monitor.AddDevFreqResource(static_cast<DeviceFlag>(i),
                                             DevFreqFlag::MIN_FREQ),
                  absl::OkStatus());
        EXPECT_EQ(monitor.AddDevFreqResource(static_cast<DeviceFlag>(i),
                                             DevFreqFlag::MAX_FREQ),
                  absl::OkStatus());
        EXPECT_EQ(monitor.AddDevFreqResource(static_cast<DeviceFlag>(i),
                                             DevFreqFlag::POLLING_INTERVAL),
                  absl::OkStatus());
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  std::ifstream log_file(log_path);
  // check if the log file is created and not empty
  EXPECT_TRUE(log_file.is_open() && log_file.good());
  // get all the lines
  std::string line;
  std::vector<std::string> lines;
  while (std::getline(log_file, line)) {
    lines.push_back(line);
  }
  // check if the log file is not empty
  EXPECT_TRUE(lines.size() > 0);
#endif  // BAND_IS_MOBILE
}

TEST(ResourceMonitorTest, GetPowerSupplyTest) {
  ResourceMonitorConfig config{"", {}, 10};
  ResourceMonitor monitor;
  // EXPECT_EQ(monitor.Init(config), absl::OkStatus());
  auto status = monitor.Init(config);

#if BAND_IS_MOBILE
  for (size_t i = 0; i < EnumLength<PowerSupplyMaskFlag>(); i++) {
    const PowerSupplyMaskFlag power_supply_type = static_cast<PowerSupplyMaskFlag>(i);
    for (size_t j = 0; j < EnumLength<PowerSupplyFlag>(); j++) {
      const PowerSupplyFlag power_supply_flag = static_cast<PowerSupplyFlag>(j);
      EXPECT_EQ(monitor.AddPowerSupplyResource(
        power_supply_type, power_supply_flag), absl::OkStatus());
    }
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(22));

  for (size_t i = 0; i < EnumLength<PowerSupplyMaskFlag>(); i++) {
    const PowerSupplyMaskFlag power_supply_type = static_cast<PowerSupplyMaskFlag>(i);
    for (size_t j = 0; j < EnumLength<PowerSupplyFlag>(); j++) {
      const PowerSupplyFlag power_supply_flag = static_cast<PowerSupplyFlag>(j);
      auto power_supply = monitor.GetPowerSupply(power_supply_type, power_supply_flag);
      EXPECT_TRUE(power_supply.ok());
      std::cout << "PowerSupply " << ToString(power_supply_type) << " "
                << ToString(power_supply_flag) << " : " << power_supply.value()
                << std::endl;
    }
  }
#endif
}

}  // namespace test
}  // namespace band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}