#include "band/device/generic.h"

#include <cstring>

#include "band/device/util.h"

#if BAND_SUPPORT_DEVICE
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace band {
namespace generic {

std::vector<std::string> GetPaths(DeviceFlag device_flag, std::string suffix) {
  std::vector<std::string> device_paths;
#if BAND_SUPPORT_DEVICE
  // TODO: Add more device-specific path
  if (device_flag == DeviceFlag::kNPU) {
    device_paths = {
        "/sys/devices/platform/17000060.devfreq_npu/devfreq/"
        "17000060.devfreq_npu/"  // Galaxy S21
    };
  } else if (device_flag == DeviceFlag::kDSP) {
    device_paths = {
        "/sys/devices/platform/soc/soc:qcom,cdsp-cdsp-l3-lat/devfreq/"
        "soc:qcom,cdsp-cdsp-l3-lat/"  // Pixel 4 Hexagon DSP
    };
  }
#endif
  for (size_t i = 0; i < device_paths.size(); i++) {
    device_paths[i] += suffix;
  }
  return device_paths;
}

int GetMinFrequencyKhz(DeviceFlag device_flag) {
#if BAND_SUPPORT_DEVICE
  return TryReadInt(GetPaths(device_flag, "min_freq")) / 1000;
#else
  return -1;
#endif
}

int GetMaxFrequencyKhz(DeviceFlag device_flag) {
#if BAND_SUPPORT_DEVICE
  return TryReadInt(GetPaths(device_flag, "max_freq")) / 1000;
#else
  return -1;
#endif
}

int GetFrequencyKhz(DeviceFlag device_flag) {
#if BAND_SUPPORT_DEVICE
  return TryReadInt(GetPaths(device_flag, "cur_freq")) / 1000;
#else
  return -1;
#endif
}

int GetTargetFrequencyKhz(DeviceFlag device_flag) {
#if BAND_SUPPORT_DEVICE
  return TryReadInt(GetPaths(device_flag, "target_freq")) / 1000;
#else
  return -1;
#endif
}

int GetPollingIntervalMs(DeviceFlag device_flag) {
#if BAND_SUPPORT_DEVICE
  return TryReadInt(GetPaths(device_flag, "polling_interval"));
#else
  return -1;
#endif
}

std::vector<int> GetAvailableFrequenciesKhz(DeviceFlag device_flag) {
  std::vector<int> frequencies;  // hz
#if BAND_SUPPORT_DEVICE
  frequencies = TryReadInts(GetPaths(device_flag, "available_frequencies"));
  for (size_t i = 0; i < frequencies.size(); i++) {
    frequencies[i] /= 1000;
  }
#endif
  return frequencies;
}

std::vector<std::pair<int, int>> GetClockStats(DeviceFlag device_flag) {
  std::vector<std::pair<int, int>> frequency_stats;

#if BAND_SUPPORT_DEVICE
  std::vector<int> frequencies = GetAvailableFrequenciesKhz(device_flag);
  std::vector<int> clock_stats =
      TryReadInts(GetPaths(device_flag, "time_in_state"));

  frequency_stats.resize(frequencies.size());
  for (size_t i = 0; i < frequency_stats.size(); i++) {
    frequency_stats[i] = {frequencies[i], clock_stats[i]};
  }
#endif
  return frequency_stats;
}

}  // namespace generic
}  // namespace band
