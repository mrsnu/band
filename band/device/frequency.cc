#include "band/device/frequency.h"

#include <map>

#include "absl/strings/str_format.h"
#include "band/common.h"
#include "band/config.h"
#include "band/device/util.h"

namespace band {

namespace {

std::string GetFreqPath(std::string path) {
  return absl::StrFormat("%s/cur_freq", path);
}

}  // anonymous namespace

Frequency::Frequency(DeviceConfig config) {
  if (config.cpu_freq_path != "" && !CheckFrequency(config.cpu_freq_path)) {
    freq_device_map_[DeviceFlag::kCPU] = config.cpu_freq_path;
  }

  if (config.gpu_freq_path != "" && !CheckFrequency(config.gpu_freq_path)) {
    freq_device_map_[DeviceFlag::kGPU] = config.gpu_freq_path;
  }

  if (config.dsp_freq_path != "" && !CheckFrequency(config.dsp_freq_path)) {
    freq_device_map_[DeviceFlag::kDSP] = config.dsp_freq_path;
  }

  if (config.npu_freq_path != "" && !CheckFrequency(config.npu_freq_path)) {
    freq_device_map_[DeviceFlag::kNPU] = config.npu_freq_path;
  }
}

size_t Frequency::GetFrequency(DeviceFlag device_flag) {
  auto path = freq_device_map_[device_flag];
  return device::TryReadSizeT({path}).value();
}

FreqInfo Frequency::GetAllFrequency() {
  std::map<DeviceFlag, size_t> freq_map;
  for (auto& pair : freq_device_map_) {
    freq_map[pair.first] = GetFrequency(pair.first);
  }
  return freq_map;
}

bool Frequency::CheckFrequency(std::string path) {
  return device::IsFileAvailable(GetFreqPath(path));
}

}  // namespace band