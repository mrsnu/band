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

#include <iostream>

#include "absl/strings/str_format.h"
#include "band/logger.h"
#include "resource_monitor.h"

namespace band {

namespace {

template <typename T>
absl::StatusOr<T> TryRead(std::vector<std::string> paths,
                          std::vector<float> multipliers = {}) {
  // get from path and multiply by multiplier
  if (multipliers.size() == 0) {
    multipliers.resize(paths.size(), 1.f);
  }

  if (paths.size() != multipliers.size()) {
    return absl::InternalError(
        "Number of paths and multipliers must be the same.");
  }

  for (size_t i = 0; i < paths.size(); i++) {
    auto path = paths[i];
    std::fstream fs(path, std::fstream::in);
    if (fs.is_open()) {
      T output;
      fs >> output;
      return output * multipliers[i];
    }
  }
  return absl::NotFoundError("No available path");
}

absl::StatusOr<size_t> TryReadSizeT(std::vector<std::string> paths,
                                    std::vector<float> multipliers = {}) {
  return TryRead<size_t>(paths, multipliers);
}

absl::StatusOr<std::vector<size_t>> TryReadSizeTs(
    std::vector<std::string> paths, std::vector<float> multipliers = {}) {
  // get from path and multiply by multiplier
  if (multipliers.size() == 0) {
    multipliers.resize(paths.size(), 1.f);
  }

  if (paths.size() != multipliers.size()) {
    return absl::InternalError(
        "Number of paths and multipliers must be the same.");
  }

  for (size_t i = 0; i < paths.size(); i++) {
    auto path = paths[i];
    std::fstream fs(path, std::fstream::in);
    if (fs.is_open()) {
      std::vector<size_t> outputs;
      size_t output;
      while (fs >> output) {
        outputs.push_back(output * multipliers[i]);
      }
      return outputs;
    }
  }
  return absl::NotFoundError("No available path");
}
}  // anonymous namespace

template <>
size_t EnumLength<ThermalFlag>() {
  return static_cast<size_t>(ThermalFlag::TZ_TEMPERATURE) + 1;
}

template <>
size_t EnumLength<DevFreqFlag>() {
  return static_cast<size_t>(DevFreqFlag::POLLING_INTERVAL) + 1;
}

template <>
size_t EnumLength<CpuFreqFlag>() {
  return static_cast<size_t>(CpuFreqFlag::TRANSITION_COUNT) + 1;
}

template <>
const char* ToString<ThermalFlag>(ThermalFlag flag) {
  switch (flag) {
    case ThermalFlag::TZ_TEMPERATURE:
      return "TZ_TEMPERATURE";
    default:
      return "UNKNOWN";
  }
}

template <>
const char* ToString<DevFreqFlag>(DevFreqFlag flag) {
  switch (flag) {
    case DevFreqFlag::CUR_FREQ:
      return "CUR_FREQ";
    case DevFreqFlag::TARGET_FREQ:
      return "TARGET_FREQ";
    case DevFreqFlag::MIN_FREQ:
      return "MIN_FREQ";
    case DevFreqFlag::MAX_FREQ:
      return "MAX_FREQ";
    case DevFreqFlag::POLLING_INTERVAL:
      return "POLLING_INTERVAL";
    default:
      return "UNKNOWN";
  }
}

template <>
const char* ToString<CpuFreqFlag>(CpuFreqFlag flag) {
  switch (flag) {
    case CpuFreqFlag::CUR_FREQ:
      return "CUR_FREQ";
    case CpuFreqFlag::TARGET_FREQ:
      return "TARGET_FREQ";
    case CpuFreqFlag::MIN_FREQ:
      return "MIN_FREQ";
    case CpuFreqFlag::MAX_FREQ:
      return "MAX_FREQ";
    case CpuFreqFlag::UP_TRANSITION_LATENCY:
      return "UP_TRANSITION_LATENCY";
    case CpuFreqFlag::DOWN_TRANSITION_LATENCY:
      return "DOWN_TRANSITION_LATENCY";
    case CpuFreqFlag::TRANSITION_COUNT:
      return "TRANSITION_COUNT";
    default:
      return "UNKNOWN";
  }
}

ResourceMonitor::~ResourceMonitor() {
  if (log_file_.is_open()) {
    log_file_.close();
  }
  is_monitoring_ = false;
  if (monitor_thread_.joinable()) {
    monitor_thread_.join();
  }

  if (log_file_.is_open()) {
    log_file_ << "}";
    log_file_.close();
  }
}

absl::Status ResourceMonitor::Init(const ResourceMonitorConfig& config) {
  if (config.log_path.size() > 0) {
    // remove existing log file if exists
    std::remove(config.log_path.c_str());
    // open log file and start from the beginning
    log_file_.open(config.log_path, std::ios::out);
    if (!log_file_.is_open()) {
      return absl::NotFoundError("Cannot open log file.");
    }
    log_file_ << "{";
  }

  dev_freq_paths_ = config.device_freq_paths;
  auto dev_freq_path_candidates =
      device::ListDirectoriesInPath(GetDevFreqBasePath());

  // add default dev freq paths
  std::map<DeviceFlag, std::vector<std::string>> target_keywords;
  target_keywords[DeviceFlag::kGPU] = {"kgsl-3d0",  // adreno
                                       "mali"};
  target_keywords[DeviceFlag::kDSP] = {
      "cdsp-cdsp-l3-lat"  // hexagon
  };
  target_keywords[DeviceFlag::kNPU] = {
      "devfreq_npu"  // samsung npu
  };

  for (auto& dev_freq_path_candidate : dev_freq_path_candidates) {
    for (auto& target_keyword : target_keywords) {
      for (auto& keyword : target_keyword.second) {
        if (dev_freq_path_candidate.find(keyword) != std::string::npos) {
          dev_freq_paths_[target_keyword.first] = dev_freq_path_candidate;
          BAND_LOG(LogSeverity::kInternal, "Found dev freq path for device %s: %s",
                   ToString(target_keyword.first),
                   dev_freq_path_candidate.c_str());
        }
      }
    }
  }

  for (auto it = dev_freq_paths_.begin(); it != dev_freq_paths_.end(); it++) {
    if (!device::IsFileAvailable(GetDevFreqBasePath() + it->second)) {
      return absl::NotFoundError(
          absl::StrFormat("Device frequency path %s not found.", it->second));
    }
  }

  is_monitoring_ = true;
  monitor_thread_ =
      std::thread(&ResourceMonitor::Monitor, this, config.monitor_interval_ms);

  return absl::OkStatus();
}

std::vector<std::string> ResourceMonitor::GetThermalPaths() const {
  return device::ListDirectoriesInPath(GetThermalBasePath());
}

std::vector<std::string> ResourceMonitor::GetCpuFreqPaths() const {
  std::vector<std::string> ret;
  // get all cpu freq paths from mask
  for (size_t i = 0; i < EnumLength<CPUMaskFlag>(); i++) {
    auto cpu_mask = static_cast<CPUMaskFlag>(i);
    if (cpu_mask == CPUMaskFlag::kAll) {
      continue;
    }
    auto cpu_freq_path = GetCpuFreqPath(cpu_mask);
    if (cpu_freq_path.ok()) {
      BAND_LOG_DEBUG("CPU frequency path: %s", cpu_freq_path.value().c_str());
      auto pathes = device::ListFilesInPath(cpu_freq_path.value().c_str());
      for (auto& path : pathes) {
        ret.push_back(cpu_freq_path.value() + "/" + path);
      }
    } else {
      BAND_LOG(LogSeverity::kWarning,
               "CPU frequency path for cpu set %s not found.",
               ToString(cpu_mask));
    }
  }
  return ret;
}

std::vector<std::string> ResourceMonitor::GetDevFreqPaths() const {
  std::vector<std::string> ret;
  // get all dev freq paths from mask
  for (size_t i = 0; i < EnumLength<DeviceFlag>(); i++) {
    auto device_flag = static_cast<DeviceFlag>(i);
    if (device_flag == DeviceFlag::kCPU) {
      continue;
    }
    auto dev_freq_path = GetDevFreqPath(device_flag);
    if (dev_freq_path.ok()) {
      auto pathes = device::ListFilesInPath(dev_freq_path.value().c_str());
      for (auto& path : pathes) {
        ret.push_back(dev_freq_path.value() + "/" + path);
      }
    } else {
      BAND_LOG(LogSeverity::kWarning,
               "Device frequency path for device %s not found.",
               ToString(device_flag));
    }
  }
  return ret;
}

bool ResourceMonitor::IsValidDevice(DeviceFlag flag) const {
  return dev_freq_paths_.find(flag) != dev_freq_paths_.end();
}

absl::StatusOr<size_t> ResourceMonitor::GetThermal(ThermalFlag flag,
                                                   size_t id) const {
  std::lock_guard<std::mutex> lock(head_mtx_);
  ThermalKey key{flag, id};
  if (thermal_status_[status_head_].find(key) ==
      thermal_status_[status_head_].end()) {
    return absl::InternalError(
        absl::StrFormat("Thermal for id %d not registered.", id));
  } else {
    return thermal_status_[status_head_].at(key);
  }
}

size_t ResourceMonitor::NumThermalResources(ThermalFlag flag) const {
  static std::once_flag once_flag;
  static size_t tzs_size = 0;
  static size_t cooling_device_size = 0;
  std::call_once(once_flag, [&]() {
    std::vector<std::string> ret;
    auto thermals = device::ListDirectoriesInPath(GetThermalBasePath());
    for (auto& thermal : thermals) {
      if (thermal.find("thermal_zone") != std::string::npos) {
        tzs_size++;
      } else if (thermal.find("cooling_device") != std::string::npos) {
        cooling_device_size++;
      }
    }
  });

  switch (flag) {
    case ThermalFlag::TZ_TEMPERATURE:
      return tzs_size;
    default:
      return 0;
  }
}

absl::StatusOr<size_t> ResourceMonitor::GetDevFreq(DeviceFlag device_flag,
                                                   DevFreqFlag flag) const {
  std::lock_guard<std::mutex> lock(head_mtx_);
  DevFreqKey key{flag, device_flag};
  if (dev_freq_status_[status_head_].find(key) ==
      dev_freq_status_[status_head_].end()) {
    return absl::InternalError(absl::StrFormat(
        "Device frequency for flag %s and device %s not registered.",
        ToString(flag), ToString(device_flag)));
  } else {
    return dev_freq_status_[status_head_].at(key);
  }
}

absl::StatusOr<std::vector<size_t>> ResourceMonitor::GetAvailableDevFreqs(
    DeviceFlag flag) const {
  absl::StatusOr<std::string> dev_freq_path = GetDevFreqPath(flag);
  RETURN_IF_ERROR(dev_freq_path.status());
  return TryReadSizeTs({dev_freq_path.value() + "/freq_table_mhz",
                        dev_freq_path.value() + "/dvfs_table"},
                       {1000.f, 1.f});
}

absl::StatusOr<size_t> ResourceMonitor::GetCpuFreq(CPUMaskFlag cpu_flag,
                                                   CpuFreqFlag flag) const {
  std::lock_guard<std::mutex> lock(head_mtx_);
  CpuFreqKey key{flag, cpu_flag};
  if (cpu_freq_status_[status_head_].find(key) ==
      cpu_freq_status_[status_head_].end()) {
    return absl::InternalError(absl::StrFormat(
        "CPU frequency for flag %s and cpu set %s not registered.",
        ToString(flag), ToString(cpu_flag)));
  } else {
    return cpu_freq_status_[status_head_].at(key);
  }
}

absl::StatusOr<std::vector<size_t>> ResourceMonitor::GetAvailableCpuFreqs(
    CPUMaskFlag cpu_set) const {
  absl::StatusOr<std::string> cpu_freq_path = GetCpuFreqPath(cpu_set);
  RETURN_IF_ERROR(cpu_freq_path.status());
  return TryReadSizeTs(
      {cpu_freq_path.value() + "/scaling_available_frequencies"});
}

absl::Status ResourceMonitor::AddThermalResource(ThermalFlag flag, size_t id) {
  std::lock_guard<std::mutex> lock(path_mtx_);
  ThermalKey key{flag, id};
  if (thermal_resources_.find(key) != thermal_resources_.end()) {
    return absl::InternalError(
        absl::StrFormat("Thermal resource id %d already registered.", id));
  }

  std::string base_path = GetThermalBasePath();
  std::string path;

  switch (flag) {
    case ThermalFlag::TZ_TEMPERATURE:
      path = absl::StrFormat("%s/thermal_zone%d/temp", base_path, id);
      break;
    default:
      return absl::InternalError(absl::StrFormat(
          "Thermal resource for flag %s not supported.", ToString(flag)));
  }

  if (!device::IsFileAvailable(path)) {
    return absl::NotFoundError(absl::StrFormat("Path %s not found.", path));
  }

  thermal_resources_[key] = {path, 1.f};
  // add initial value
  size_t value = TryReadSizeT({path}).value();
  thermal_status_[0][key] = value;
  thermal_status_[1][key] = value;
  return absl::OkStatus();
}

absl::Status ResourceMonitor::AddCpuFreqResource(CPUMaskFlag cpu_flag,
                                                 CpuFreqFlag flag) {
  std::lock_guard<std::mutex> lock(path_mtx_);
  CpuFreqKey key{flag, cpu_flag};
  if (cpu_freq_resources_.find(key) != cpu_freq_resources_.end()) {
    return absl::InternalError(absl::StrFormat(
        "CPU frequency resource for flag %s and cpu set %s already registered.",
        ToString(flag), ToString(cpu_flag)));
  }

  absl::StatusOr<std::string> cpu_freq_path = GetCpuFreqPath(cpu_flag);
  RETURN_IF_ERROR(cpu_freq_path.status());
  std::string base_path = cpu_freq_path.value();

  std::vector<std::string> path_candidates;
  std::vector<float> multipliers;
  bool require_continuous_monitoring = true;
  switch (flag) {
    case CpuFreqFlag::CUR_FREQ:
      path_candidates = {base_path + "/cpuinfo_cur_freq",
                         base_path + "/scaling_cur_freq"};
      break;
    case CpuFreqFlag::TARGET_FREQ:
      path_candidates = {base_path + "/scaling_cur_freq"};
      break;
    case CpuFreqFlag::MAX_FREQ:
      path_candidates = {base_path + "/scaling_max_freq"};
      break;
    case CpuFreqFlag::MIN_FREQ:
      path_candidates = {base_path + "/scaling_min_freq"};
      break;
    case CpuFreqFlag::UP_TRANSITION_LATENCY:
      require_continuous_monitoring = false;
      path_candidates = {base_path + "/schedutil/up_rate_limit_us",
                         base_path + "/cpuinfo_transition_latency"};
      multipliers = {1.f, 0.001f};
      break;
    case CpuFreqFlag::DOWN_TRANSITION_LATENCY:
      require_continuous_monitoring = false;
      path_candidates = {base_path + "/schedutil/down_rate_limit_us",
                         base_path + "/cpuinfo_transition_latency"};
      multipliers = {1.f, 0.001f};
      break;
    case CpuFreqFlag::TRANSITION_COUNT:
      require_continuous_monitoring = false;
      path_candidates = {base_path + "/stats/total_trans"};
      break;
    default:
      return absl::InternalError(absl::StrFormat(
          "CPU frequency resource for flag %s not supported.", ToString(flag)));
  }

  if (require_continuous_monitoring) {
    absl::StatusOr<std::string> path = GetFirstAvailablePath(path_candidates);
    RETURN_IF_ERROR(path.status());
    // all cpu freqs are in kHz, requires no conversion
    cpu_freq_resources_[key] = {path.value(), 1.f};
  } else {
    std::lock_guard<std::mutex> lock(head_mtx_);
    auto value = TryReadSizeT(path_candidates, multipliers);
    RETURN_IF_ERROR(value.status());
    cpu_freq_status_[0][key] = value.value();
    cpu_freq_status_[1][key] = value.value();
  }

  return absl::OkStatus();
}  // namespace band

absl::Status ResourceMonitor::AddDevFreqResource(DeviceFlag device_flag,
                                                 DevFreqFlag flag) {
  std::lock_guard<std::mutex> lock(path_mtx_);
  DevFreqKey key{flag, device_flag};
  if (dev_freq_resources_.find(key) != dev_freq_resources_.end()) {
    return absl::InternalError(absl::StrFormat(
        "Device frequency resource for flag %s and device %s already "
        "registered.",
        ToString(flag), ToString(device_flag)));
  }

  absl::StatusOr<std::string> dev_freq_path = GetDevFreqPath(device_flag);
  RETURN_IF_ERROR(dev_freq_path.status());
  std::string base_path = dev_freq_path.value();

  std::vector<std::string> path_candidates;
  std::vector<float> multipliers = {};
  bool require_continuous_monitoring = true;
  switch (flag) {
    case DevFreqFlag::CUR_FREQ:
      path_candidates = {base_path + "/cur_freq", base_path + "/target_freq"};
      break;
    case DevFreqFlag::TARGET_FREQ:
      path_candidates = {base_path + "/target_freq"};
      break;
    case DevFreqFlag::MAX_FREQ:
      path_candidates = {base_path + "/max_freq"};
      break;
    case DevFreqFlag::MIN_FREQ:
      path_candidates = {base_path + "/min_freq"};
      break;
    case DevFreqFlag::POLLING_INTERVAL: {
      require_continuous_monitoring = false;
      path_candidates = {base_path + "/polling_interval"};
      // ms to us
      multipliers = {1000.f};
    } break;
    default:
      return absl::InternalError(absl::StrFormat(
          "Device frequency resource for flag %s not supported.",
          ToString(flag)));
  }

  if (require_continuous_monitoring) {
    absl::StatusOr<std::string> path = GetFirstAvailablePath(path_candidates);
    RETURN_IF_ERROR(path.status());
    // all dev freqs are in Hz should be converted to kHz
    dev_freq_resources_[key] = {path.value(), 0.001f};
  } else {
    std::lock_guard<std::mutex> lock(head_mtx_);
    auto value = TryReadSizeT(path_candidates, multipliers);
    RETURN_IF_ERROR(value.status());
    dev_freq_status_[0][key] = value.value();
    dev_freq_status_[1][key] = value.value();
  }
  return absl::OkStatus();
}

absl::Status ResourceMonitor::AddNetworkResource(NetworkFlag) {
  return absl::Status();
}

void ResourceMonitor::AddOnUpdate(
    std::function<void(const ResourceMonitor&)> callback) {
  std::lock_guard<std::mutex> lock(callback_mtx_);
  on_update_callbacks_.push_back(callback);
}

const char* ResourceMonitor::GetThermalBasePath() {
  return "/sys/class/thermal/";
}

const char* ResourceMonitor::GetCpuFreqBasePath() {
  return "/sys/devices/system/cpu/cpufreq/";
}

const char* ResourceMonitor::GetDevFreqBasePath() {
  return "/sys/class/devfreq/";
}

absl::StatusOr<std::string> ResourceMonitor::GetDevFreqPath(
    DeviceFlag flag) const {
  if (dev_freq_paths_.find(flag) == dev_freq_paths_.end()) {
    return absl::InternalError("Dev frequency resource not registered.");
  } else {
    return GetDevFreqBasePath() + dev_freq_paths_.at(flag);
  }
}

absl::StatusOr<std::string> ResourceMonitor::GetCpuFreqPath(
    CPUMaskFlag flag) const {
  static std::map<CPUMaskFlag, std::string> cpu_freq_paths;
  static std::once_flag once_flag;
  std::call_once(once_flag, [&]() {
    auto cpu_freqs = device::ListDirectoriesInPath(GetCpuFreqBasePath());

    std::map<size_t, CPUMaskFlag> representative_cpu_ids;
    for (size_t i = 0; i < EnumLength<CPUMaskFlag>(); i++) {
      if (i == static_cast<size_t>(CPUMaskFlag::kAll)) {
        continue;
      }
      const CPUMaskFlag flag = static_cast<CPUMaskFlag>(i);
      CpuSet cpu_set = BandCPUMaskGetSet(flag);
      // get first cpu id in the set
      for (size_t j = 0; j < GetCPUCount(); j++) {
        if (cpu_set.IsEnabled(j)) {
          representative_cpu_ids[j] = flag;
          break;
        }
      }
    }

    for (auto& cpu_freq : cpu_freqs) {
      if (cpu_freq.find("policy") == std::string::npos) {
        continue;
      }

      // get representative cpu id from the integer after "policy"
      size_t id = std::stoi(cpu_freq.substr(6));
      if (representative_cpu_ids.find(id) != representative_cpu_ids.end()) {
        cpu_freq_paths[representative_cpu_ids.at(id)] = cpu_freq;
      }
    }
  });

  if (cpu_freq_paths.find(flag) == cpu_freq_paths.end()) {
    return absl::InternalError(absl::StrFormat(
        "CPU frequency for flag %s not found.", ToString(flag)));
  } else {
    return GetCpuFreqBasePath() + cpu_freq_paths.at(flag);
  }
}

absl::StatusOr<std::string> ResourceMonitor::GetFirstAvailablePath(
    const std::vector<std::string>& paths) const {
  for (auto& path : paths) {
    if (device::IsFileAvailable(path) && TryReadSizeT({path}).ok()) {
      return path;
    }
  }

  return absl::NotFoundError("No available path");
}
void ResourceMonitor::Monitor(size_t interval_ms) {
  while (is_monitoring_) {
    // measure time to monitor and sleep for the rest of the interval
    auto start = std::chrono::high_resolution_clock::now();
    // update resource status
    size_t next_head = (status_head_ + 1) % 2;
    {
      std::lock_guard<std::mutex> lock(path_mtx_);
      // thermal
      for (auto& resource : thermal_resources_) {
        absl::StatusOr<size_t> status = TryReadSizeT({resource.second.first});
        if (status.ok()) {
          thermal_status_[next_head][resource.first] =
              status.value() * resource.second.second;
        } else {
          BAND_LOG(LogSeverity::kWarning,
                   "Failed to read thermal resource %s (%s, %d): %s",
                   ToString(resource.first.first),
                   resource.second.first.c_str(), resource.first.second,
                   status.status().ToString().c_str());
        }
      }

      // cpu freq
      for (auto& resource : cpu_freq_resources_) {
        absl::StatusOr<size_t> status = TryReadSizeT({resource.second.first});
        if (status.ok()) {
          cpu_freq_status_[next_head][resource.first] =
              status.value() * resource.second.second;
        } else {
          BAND_LOG(LogSeverity::kWarning,
                   "Failed to read cpu freq resource %s: %s",
                   resource.second.first.c_str(),
                   status.status().ToString().c_str());
        }
      }

      // dev freq
      for (auto& resource : dev_freq_resources_) {
        absl::StatusOr<size_t> status = TryReadSizeT({resource.second.first});
        if (status.ok()) {
          dev_freq_status_[next_head][resource.first] =
              status.value() * resource.second.second;
        } else {
          BAND_LOG(LogSeverity::kWarning,
                   "Failed to read dev freq resource %s: %s",
                   resource.second.first.c_str(),
                   status.status().ToString().c_str());
        }
      }
    }

    // swap head
    {
      std::lock_guard<std::mutex> lock(head_mtx_);
      status_head_ = next_head;
    }

    // report
    {
      std::lock_guard<std::mutex> lock(callback_mtx_);
      for (auto& callback : on_update_callbacks_) {
        callback(*this);
      }
    }

    // log to file
    if (log_file_.is_open()) {
      log_file_ << "{\"thermal\": {";
      for (auto& resource : thermal_status_[status_head_]) {
        log_file_ << "\"" << ToString(resource.first.first) << "_"
                  << resource.first.second << "\": " << resource.second << ", ";
      }
      log_file_ << "}, \"cpu_freq\": {";
      for (auto& resource : cpu_freq_status_[status_head_]) {
        log_file_ << "\"" << ToString(resource.first.first) << "_"
                  << ToString(resource.first.second)
                  << "\": " << resource.second << ", ";
      }
      log_file_ << "}, \"dev_freq\": {";
      for (auto& resource : dev_freq_status_[status_head_]) {
        log_file_ << "\"" << ToString(resource.first.first) << "_"
                  << ToString(resource.first.second)
                  << "\": " << resource.second << ", ";
      }
      log_file_ << "}},";
    }

    std::this_thread::sleep_for(
        std::chrono::milliseconds(interval_ms) -
        (std::chrono::high_resolution_clock::now() - start));
  }
}
}  // namespace band