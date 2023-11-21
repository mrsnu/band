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

std::string GetCpuAvailableFreqPath(const std::string& path) {
  return absl::StrFormat("%s/scaling_available_frequencies", path.c_str());
}

std::string GetGpuFreqPath(const std::string& path) {
  return absl::StrFormat("%s/gpuclk", path.c_str());
}

std::string GetGpuMinScalingPath(const std::string& path) {
  return absl::StrFormat("%s/min_pwrlevel", path.c_str());
}

std::string GetGpuMaxScalingPath(const std::string& path) {
  return absl::StrFormat("%s/max_pwrlevel", path.c_str());
}

std::string GetGpuAvailableFreqPath(const std::string& path) {
  return absl::StrFormat("%s/devfreq/available_frequencies", path.c_str());
}

}  // anonymous namespace

Frequency::Frequency(DeviceConfig config) : config_(config) {
  device::Root();

  if (config.cpu_freq_path != "" && CheckFrequency(config.cpu_freq_path)) {
    freq_device_map_[FreqFlag::kCPU] = config.cpu_freq_path;
  } else {
    BAND_LOG_PROD(BAND_LOG_ERROR, "CPU frequency path \"%s\" is not available.",
                  config.cpu_freq_path.c_str());
  }

  if (config.gpu_freq_path != "" && CheckFrequency(config.gpu_freq_path)) {
    freq_device_map_[FreqFlag::kGPU] = config.gpu_freq_path;
  } else {
    BAND_LOG_PROD(BAND_LOG_ERROR, "GPU frequency path \"%s\" is not available.",
                  config.gpu_freq_path.c_str());
  }

  if (config.runtime_freq_path != "" &&
      CheckFrequency(config.runtime_freq_path)) {
    freq_device_map_[FreqFlag::kRuntime] = config.runtime_freq_path;
  } else {
    BAND_LOG_PROD(BAND_LOG_ERROR,
                  "Runtime CPU frequency path \"%s\" is not available.",
                  config.runtime_freq_path.c_str());
  }
}

double Frequency::GetFrequency(FreqFlag device_flag) {
  auto path = freq_device_map_[device_flag];
  if (device_flag == FreqFlag::kCPU) {
    return device::TryReadDouble({GetCpuFreqPath(path)}, {cpu_freq_multiplier})
        .value();
  }
  return device::TryReadDouble({GetGpuFreqPath(path)}, {dev_freq_multiplier})
      .value();
}

absl::Status Frequency::SetRuntimeFrequency(double freq) {
  if (freq_device_map_.find(FreqFlag::kRuntime) == freq_device_map_.end()) {
    return absl::InternalError("Runtime CPU frequency path is not available.");
  }
  return SetFrequencyWithPath(
      GetCpuScalingPath(freq_device_map_.at(FreqFlag::kRuntime)), freq,
      cpu_freq_multiplier_w);
}

absl::Status Frequency::SetCpuFrequency(double freq) {
  if (freq_device_map_.find(FreqFlag::kCPU) == freq_device_map_.end()) {
    return absl::InternalError("CPU frequency path is not available.");
  }
  return SetFrequencyWithPath(
      GetCpuScalingPath(freq_device_map_.at(FreqFlag::kCPU)), freq,
      cpu_freq_multiplier_w);
}

absl::Status Frequency::SetGpuFrequency(double freq) {
  if (freq_device_map_.find(FreqFlag::kGPU) == freq_device_map_.end()) {
    return absl::InternalError("GPU frequency path is not available.");
  }
  auto status1 = device::TryWriteSizeT(
      {GetGpuMinScalingPath(freq_device_map_.at(FreqFlag::kGPU))},
      gpu_freq_map_.at(static_cast<size_t>(freq * dev_freq_multiplier_w)));
  auto status2 = device::TryWriteSizeT(
      {GetGpuMaxScalingPath(freq_device_map_.at(FreqFlag::kGPU))},
      gpu_freq_map_.at(static_cast<size_t>(freq * dev_freq_multiplier_w)));
  auto status3 = device::TryWriteSizeT(
      {GetGpuMinScalingPath(freq_device_map_.at(FreqFlag::kGPU))},
      gpu_freq_map_.at(static_cast<size_t>(freq * dev_freq_multiplier_w)));
  auto status4 = device::TryWriteSizeT(
      {GetGpuMaxScalingPath(freq_device_map_.at(FreqFlag::kGPU))},
      gpu_freq_map_.at(static_cast<size_t>(freq * dev_freq_multiplier_w)));
  if (!status1.ok() || !status2.ok() || !status3.ok() || !status4.ok()) {
    return absl::InternalError("Failed to set GPU frequency.");
  }
  return absl::OkStatus();
}

absl::Status Frequency::SetFrequencyWithPath(const std::string& path,
                                             double freq, size_t multiplier) {
  return device::TryWriteSizeT({path}, static_cast<size_t>(freq * multiplier));
}

FreqMap Frequency::GetAllFrequency() {
  std::map<FreqFlag, double> freq_map;
  for (auto& pair : freq_device_map_) {
    freq_map[pair.first] = GetFrequency(pair.first);
  }
  return freq_map;
}

std::map<FreqFlag, std::vector<double>> Frequency::GetAllAvailableFrequency() {
  if (freq_available_map_.size() > 0) {
    return freq_available_map_;
  }

  std::map<FreqFlag, std::vector<double>> freq_map;
  for (auto& pair : freq_device_map_) {
    auto path = pair.second;
    if (pair.first == FreqFlag::kCPU) {
      auto freqs = device::TryReadDoubles({GetCpuAvailableFreqPath(path)},
                                          {cpu_freq_multiplier})
                       .value();
      freq_map[pair.first] = freqs;
    } else {
      auto freqs = device::TryReadDoubles({GetGpuAvailableFreqPath(path)},
                                          {dev_freq_multiplier})
                       .value();
      freq_map[pair.first] = freqs;
    }
  }
  freq_available_map_ = freq_map;
  return freq_available_map_;
}

bool Frequency::CheckFrequency(std::string path) {
  return device::IsFileAvailable(path);
}

}  // namespace band