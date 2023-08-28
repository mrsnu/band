/*
 * Copyright 2023 Seoul National University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef BAND_RESOURCE_MONITOR_H_
#define BAND_RESOURCE_MONITOR_H_

#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "band/common.h"
#include "band/config.h"
#include "band/device/cpu.h"

namespace band {

enum class ThermalFlag {
  TZ_TEMPERATURE = 0,
};

// NOTE:
// All frequency values are in KHz
// All latency values are in us
enum class DevFreqFlag {
  CUR_FREQ = 0,
  TARGET_FREQ = 1,
  MIN_FREQ = 2,
  MAX_FREQ = 3,
  POLLING_INTERVAL = 4,
};

enum class CpuFreqFlag {
  CUR_FREQ = 0,
  TARGET_FREQ = 1,
  MIN_FREQ = 2,
  MAX_FREQ = 3,
  UP_TRANSITION_LATENCY = 4,
  DOWN_TRANSITION_LATENCY = 5,
  TRANSITION_COUNT = 6,
};

enum class NetworkFlag {};

template <>
size_t EnumLength<ThermalFlag>();
template <>
size_t EnumLength<DevFreqFlag>();
template <>
size_t EnumLength<CpuFreqFlag>();

template <>
const char* ToString<ThermalFlag>(ThermalFlag flag);
template <>
const char* ToString<DevFreqFlag>(DevFreqFlag flag);
template <>
const char* ToString<CpuFreqFlag>(CpuFreqFlag flag);

class ResourceMonitor {
 public:
  ~ResourceMonitor();
  ResourceMonitor() = default;

  absl::Status Init(const ResourceMonitorConfig& config);

  // for debugging. print out all the paths found
  std::vector<std::string> GetThermalPaths() const;
  std::vector<std::string> GetCpuFreqPaths() const;
  std::vector<std::string> GetDevFreqPaths() const;

  // check if the device is valid (after Init() is called)
  bool IsValidDevice(DeviceFlag flag) const;

  // get thermal resource (thermal zone or cooling device)
  absl::StatusOr<size_t> GetThermal(ThermalFlag flag, size_t id = 0) const;
  // get number of thermal resources corresponding to the flag
  size_t NumThermalResources(ThermalFlag flag) const;
  absl::StatusOr<size_t> GetDevFreq(DeviceFlag device_flag,
                                    DevFreqFlag flag) const;
  absl::StatusOr<std::vector<size_t>> GetAvailableDevFreqs(
      DeviceFlag flag) const;
  absl::StatusOr<size_t> GetCpuFreq(CPUMaskFlag cpu_flag,
                                    CpuFreqFlag flag) const;
  absl::StatusOr<std::vector<size_t>> GetAvailableCpuFreqs(
      CPUMaskFlag cpu_flag) const;

  // register target resource to monitor
  absl::Status AddThermalResource(ThermalFlag flag, size_t id);
  absl::Status AddCpuFreqResource(CPUMaskFlag cpu_flag, CpuFreqFlag flag);
  absl::Status AddDevFreqResource(DeviceFlag device_flag, DevFreqFlag flag);
  absl::Status AddNetworkResource(NetworkFlag flag);

  // add listener to the resource update
  void AddOnUpdate(std::function<void(const ResourceMonitor&)> callback);

 private:
  static const char* GetThermalBasePath();
  static const char* GetCpuFreqBasePath();
  static const char* GetDevFreqBasePath();

  std::map<DeviceFlag, std::string> dev_freq_paths_;
  absl::StatusOr<std::string> GetDevFreqPath(DeviceFlag flag) const;
  absl::StatusOr<std::string> GetCpuFreqPath(CPUMaskFlag flag) const;

  // Read from the first available path, return absl::NotFoundError if none of
  // the paths exist.
  absl::StatusOr<std::string> GetFirstAvailablePath(
      const std::vector<std::string>& paths) const;

  std::thread monitor_thread_;
  bool is_monitoring_ = false;
  // main monitoring loop
  void Monitor(size_t interval_ms);

  mutable std::ofstream log_file_;

  using ThermalKey = std::pair<ThermalFlag, size_t>;
  using CpuFreqKey = std::pair<CpuFreqFlag, CPUMaskFlag>;
  using DevFreqKey = std::pair<DevFreqFlag, DeviceFlag>;

  mutable std::mutex path_mtx_;
  // registered thermal resources
  // (flag, multiplier)
  std::map<ThermalKey, std::pair<std::string, float>> thermal_resources_;
  std::map<CpuFreqKey, std::pair<std::string, float>> cpu_freq_resources_;
  std::map<DevFreqKey, std::pair<std::string, float>> dev_freq_resources_;

  mutable std::mutex head_mtx_;
  size_t status_head_ = 0;
  std::array<std::map<ThermalKey, size_t>, 2> thermal_status_;
  std::array<std::map<CpuFreqKey, size_t>, 2> cpu_freq_status_;
  std::array<std::map<DevFreqKey, size_t>, 2> dev_freq_status_;

  std::mutex callback_mtx_;
  std::vector<std::function<void(const ResourceMonitor&)>> on_update_callbacks_;
};

}  // namespace band

#endif  // BAND_RESOURCE_MONITOR_H_