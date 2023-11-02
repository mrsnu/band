#include "band/device/frequency.h"

#include <map>

#include "absl/strings/str_format.h"
#include "band/device/util.h"
#include "band/logger.h"

namespace band {

namespace {

std::string GetCpuFreqPath(const std::string& path) {
  return absl::StrFormat("%s/scaling_cur_freq", path.c_str());
}

std::string GetCpuScalingPath(const std::string& path) {
  return absl::StrFormat("%s/scaling_setspeed", path.c_str());
}

std::string GetFreqPath(const std::string& path) {
  return absl::StrFormat("%s/cur_freq", path.c_str());
}

std::string GetScalingPath(const std::string& path) {
  return absl::StrFormat("%s/userspace/set_freq", path.c_str());
}

std::string GetCpuAvailableFreqPath(const std::string& path) {
  return absl::StrFormat("%s/scaling_available_frequencies", path.c_str());
}

std::string GetAvailableFreqPath(const std::string& path) {
  return absl::StrFormat("%s/available_frequencies", path.c_str());
}

}  // anonymous namespace

Frequency::Frequency(DeviceConfig config) : config_(config) {
  device::Root();

  if (config.runtime_freq_path != "" &&
      CheckFrequency(config.runtime_freq_path)) {
    runtime_cpu_path_ = config.runtime_freq_path;
  } else {
    BAND_LOG_PROD(BAND_LOG_ERROR,
                  "Runtime frequency path \"%s\" is not available.",
                  config.cpu_freq_path.c_str());
  }

  if (config.cpu_freq_path != "" && CheckFrequency(config.cpu_freq_path)) {
    freq_device_map_[DeviceFlag::kCPU] = config.cpu_freq_path;
  } else {
    BAND_LOG_PROD(BAND_LOG_ERROR, "CPU frequency path \"%s\" is not available.",
                  config.cpu_freq_path.c_str());
  }

  if (config.gpu_freq_path != "" && CheckFrequency(config.gpu_freq_path)) {
    freq_device_map_[DeviceFlag::kGPU] = config.gpu_freq_path;
  } else {
    BAND_LOG_PROD(BAND_LOG_ERROR, "GPU frequency path \"%s\" is not available.",
                  config.gpu_freq_path.c_str());
  }
}

double Frequency::GetFrequency(DeviceFlag device_flag) {
  auto path = freq_device_map_[device_flag];
  if (device_flag == DeviceFlag::kCPU) {
    return device::TryReadDouble({GetCpuFreqPath(path)},
                                 {config_.cpu_freq_multiplier})
        .value();
  }
  return device::TryReadDouble({GetFreqPath(path)},
                               {config_.dev_freq_multiplier})
      .value();
}

double Frequency::GetRuntimeFrequency() {
  return device::TryReadDouble({GetCpuFreqPath(config_.runtime_freq_path)},
                               {config_.cpu_freq_multiplier})
      .value();
}

absl::Status Frequency::SetFrequency(DeviceFlag device_flag, double freq) {
  if (freq_available_map_.find(device_flag) == freq_available_map_.end()) {
    return absl::UnavailableError(
        "The given device is not available for DVFS.");
  }

  if (device_flag == DeviceFlag::kCPU) {
    return SetCpuFrequency(freq);
  }

  return SetDevFrequency(device_flag, freq);
}

absl::Status Frequency::SetRuntimeFrequency(double freq) {
  return SetFrequencyWithPath(GetCpuScalingPath(runtime_cpu_path_), freq,
                              config_.cpu_freq_multiplier_w);
}

absl::Status Frequency::SetCpuFrequency(double freq) {
  return SetFrequencyWithPath(
      GetCpuScalingPath(freq_device_map_.at(DeviceFlag::kCPU)), freq,
      config_.cpu_freq_multiplier_w);
}

absl::Status Frequency::SetDevFrequency(DeviceFlag device_flag, double freq) {
  if (freq_available_map_.find(device_flag) == freq_available_map_.end()) {
    return absl::UnavailableError(
        "The given device is not available for DVFS.");
  }

  return SetFrequencyWithPath(GetScalingPath(freq_device_map_.at(device_flag)),
                              freq, config_.dev_freq_multiplier_w);
}

absl::Status Frequency::SetFrequencyWithPath(const std::string& path,
                                             double freq, size_t multiplier) {
  return device::TryWriteSizeT({path}, static_cast<size_t>(freq * multiplier));
}

FreqMap Frequency::GetAllFrequency() {
  std::map<DeviceFlag, double> freq_map;
  for (auto& pair : freq_device_map_) {
    freq_map[pair.first] = GetFrequency(pair.first);
  }
  return freq_map;
}

std::map<DeviceFlag, std::vector<double>>
Frequency::GetAllAvailableFrequency() {
  if (freq_available_map_.size() > 0) {
    return freq_available_map_;
  }

  std::map<DeviceFlag, std::vector<double>> freq_map;
  for (auto& pair : freq_device_map_) {
    auto path = pair.second;
    if (pair.first == DeviceFlag::kCPU) {
      auto freqs = device::TryReadDoubles({GetCpuAvailableFreqPath(path)},
                                          {config_.cpu_freq_multiplier})
                       .value();
      freq_map[pair.first] = freqs;
    } else {
      auto freqs = device::TryReadDoubles({GetAvailableFreqPath(path)},
                                          {config_.dev_freq_multiplier})
                       .value();
      freq_map[pair.first] = freqs;
    }
  }
  freq_available_map_ = freq_map;
  return freq_available_map_;
}

std::vector<double> Frequency::GetRuntimeAvailableFrequency() {
  return device::TryReadDoubles(
             {GetCpuAvailableFreqPath(config_.runtime_freq_path)},
             {config_.cpu_freq_multiplier})
      .value();
}

bool Frequency::CheckFrequency(std::string path) {
  return device::IsFileAvailable(path);
}

}  // namespace band