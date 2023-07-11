#include "band/device/generic.h"

#include <cstring>

#include "band/device/util.h"

#if BAND_SUPPORT_DEVICE
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace band {
using namespace device;
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

absl::StatusOr<size_t> GetMinFrequencyKhz(DeviceFlag device_flag) {
#if BAND_SUPPORT_DEVICE
  absl::StatusOr<size_t> min_freq =
      TryReadSizeT(GetPaths(device_flag, "min_freq"));
  if (min_freq.ok()) {
    return min_freq.value() / 1000;
  } else {
    return min_freq.status();
  }
#else
  return absl::UnavailableError("Device not supported");
#endif
}

absl::StatusOr<size_t> GetMaxFrequencyKhz(DeviceFlag device_flag) {
#if BAND_SUPPORT_DEVICE
  absl::StatusOr<size_t> max_freq =
      TryReadSizeT(GetPaths(device_flag, "max_freq"));
  if (max_freq.ok()) {
    return max_freq.value() / 1000;
  } else {
    return max_freq.status();
  }
#else
  return absl::UnavailableError("Device not supported");
#endif
}

absl::StatusOr<size_t> GetFrequencyKhz(DeviceFlag device_flag) {
#if BAND_SUPPORT_DEVICE
  absl::StatusOr<size_t> cur_freq =
      TryReadSizeT(GetPaths(device_flag, "cur_freq"));
  if (cur_freq.ok()) {
    return cur_freq.value() / 1000;
  } else {
    return cur_freq.status();
  }
#else
  return absl::UnavailableError("Device not supported");
#endif
}

absl::StatusOr<size_t> GetTargetFrequencyKhz(DeviceFlag device_flag) {
#if BAND_SUPPORT_DEVICE
  absl::StatusOr<size_t> target_freq =
      TryReadSizeT(GetPaths(device_flag, "target_freq"));
  if (target_freq.ok()) {
    return target_freq.value() / 1000;
  } else {
    return target_freq.status();
  }
#else
  return absl::UnavailableError("Device not supported");
#endif
}

absl::StatusOr<size_t> GetPollingIntervalMs(DeviceFlag device_flag) {
#if BAND_SUPPORT_DEVICE
  return TryReadSizeT(GetPaths(device_flag, "polling_interval"));
#else
  return absl::UnavailableError("Device not supported");
#endif
}

absl::StatusOr<std::vector<size_t>> GetAvailableFrequenciesKhz(
    DeviceFlag device_flag) {
#if BAND_SUPPORT_DEVICE
  absl::StatusOr<std::vector<size_t>> frequencies;  // hz
  frequencies = TryReadSizeTs(GetPaths(device_flag, "available_frequencies"));
  if (!frequencies.ok()) {
    return frequencies.status();
  } else {
    std::vector<size_t> frequency_values = frequencies.value();
    for (size_t i = 0; i < frequency_values.size(); i++) {
      frequency_values[i] /= 1000;
    }
    return frequency_values;
  }
#endif
  return absl::UnavailableError("Device not supported");
}

absl::StatusOr<std::vector<std::pair<size_t, size_t>>> GetClockStats(
    DeviceFlag device_flag) {
#if BAND_SUPPORT_DEVICE
  std::vector<std::pair<size_t, size_t>> frequency_stats;

  absl::StatusOr<std::vector<size_t>> frequencies =
      GetAvailableFrequenciesKhz(device_flag);
  absl::StatusOr<std::vector<size_t>> clock_stats =
      TryReadSizeTs(GetPaths(device_flag, "time_in_state"));

  if (!frequencies.ok()) {
    return frequencies.status();
  } else if (!clock_stats.ok()) {
    return clock_stats.status();
  }

  frequency_stats.resize(frequencies.value().size());
  for (size_t i = 0; i < frequency_stats.size(); i++) {
    frequency_stats[i] = {frequencies.value()[i], clock_stats.value()[i]};
  }
  return frequency_stats;
#else
  return absl::UnavailableError("Device not supported");
#endif
}

}  // namespace generic
}  // namespace band
