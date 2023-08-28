#include "band/device/frequency.h"

#include <map>

#include "absl/strings/str_format.h"
#include "band/common.h"
#include "band/config.h"
#include "band/device/util.h"
#include "band/logger.h"

namespace band {

namespace {

std::string GetCpuFreqPath(std::string path) {
  return absl::StrFormat("%s/scaling_cur_freq", path.c_str());
}

std::string GetFreqPath(std::string path) {
  return absl::StrFormat("%s/cur_freq", path.c_str());
}

std::string GetCpuAvailableFreqPath(std::string path) {
  return absl::StrFormat("%s/scaling_available_frequencies", path.c_str());
}

std::string GetAvailableFreqPath(std::string path) {
  return absl::StrFormat("%s/available_frequencies", path.c_str());
}

}  // anonymous namespace

Frequency::Frequency(DeviceConfig config) {
  device::Root();

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

  if (config.dsp_freq_path != "" && CheckFrequency(config.dsp_freq_path)) {
    freq_device_map_[DeviceFlag::kDSP] = config.dsp_freq_path;
  } else {
    BAND_LOG_PROD(BAND_LOG_ERROR, "DSP frequency path \"%s\" is not available.",
                  config.dsp_freq_path.c_str());
  }

  if (config.npu_freq_path != "" && CheckFrequency(config.npu_freq_path)) {
    freq_device_map_[DeviceFlag::kNPU] = config.npu_freq_path;
  } else {
    BAND_LOG_PROD(BAND_LOG_ERROR, "NPU frequency path \"%s\" is not available.",
                  config.npu_freq_path.c_str());
  }
}

double Frequency::GetFrequency(DeviceFlag device_flag) {
  auto path = freq_device_map_[device_flag];
  if (device_flag == DeviceFlag::kCPU) {
    return device::TryReadDouble({GetCpuFreqPath(path)}, {1.0E-6f}).value();
  }
  return device::TryReadDouble({GetFreqPath(path)}, {1.0E-9f}).value();
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
      auto freqs =
          device::TryReadDoubles({GetCpuAvailableFreqPath(path)}, {1.0E-6f})
              .value();
      freq_map[pair.first] = freqs;
    } else {
      auto freqs =
          device::TryReadDoubles({GetAvailableFreqPath(path)}, {1.0E-9f})
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