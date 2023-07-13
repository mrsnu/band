#include "band/device/gpu.h"

#include <cstring>

#include "band/device/util.h"

#if BAND_IS_MOBILE
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace band {
using namespace device;
namespace gpu {

std::vector<std::string> GetPaths(std::string suffix) {
  std::vector<std::string> device_paths;
#if BAND_IS_MOBILE
  // TODO: Add more device-specific GPU path
  device_paths = {
      "/sys/class/kgsl/kgsl-3d0/",     // Pixel4
      "/sys/class/misc/mali0/device/"  // Galaxy S21 (Mali)
  };
#endif
  for (size_t i = 0; i < device_paths.size(); i++) {
    device_paths[i] += suffix;
  }
  return device_paths;
}

absl::StatusOr<size_t> GetMinFrequencyKhz() {
#if BAND_IS_MOBILE
  auto min_freq = TryReadSizeT(GetPaths("min_clock_mhz"));
  if (min_freq.ok()) {
    return min_freq.value() * 1000;
  } else {
    return min_freq.status();
  }
#else
  return absl::UnavailableError("Device not supported");
#endif
}

absl::StatusOr<size_t> GetMaxFrequencyKhz() {
#if BAND_IS_MOBILE
  auto max_freq = TryReadSizeT(GetPaths("max_clock_mhz"));
  if (max_freq.ok()) {
    return max_freq.value() * 1000;
  } else {
    return max_freq.status();
  }
#else
  return absl::UnavailableError("Device not supported");
#endif
}

absl::StatusOr<size_t> GetFrequencyKhz() {
#if BAND_IS_MOBILE
  auto freq = TryReadSizeT(GetPaths("clock_mhz"));
  if (freq.ok()) {
    return freq.value() * 1000;
  } else {
    return freq.status();
  }
#else
  return absl::UnavailableError("Device not supported");
#endif
}

absl::StatusOr<size_t> GetPollingIntervalMs() {
#if BAND_IS_MOBILE
  return TryReadSizeT(GetPaths("devfreq/polling_interval"));
#else
  return absl::UnavailableError("Device not supported");
#endif
}

absl::StatusOr<std::vector<size_t>> GetAvailableFrequenciesKhz() {
#if BAND_IS_MOBILE
  absl::StatusOr<std::vector<size_t>> frequenciesMhz;
  frequenciesMhz = TryReadSizeTs(GetPaths("freq_table_mhz"));
  if (!frequenciesMhz.ok() || frequenciesMhz.value().empty()) {
    frequenciesMhz = TryReadSizeTs(GetPaths("dvfs_table"));
  }
  for (size_t i = 0; i < frequenciesMhz.value().size(); i++) {
    frequenciesMhz.value()[i] *= 1000;
  }
  return frequenciesMhz;
#else
  return absl::UnavailableError("Device not supported");
#endif
}

absl::StatusOr<std::vector<std::pair<size_t, size_t>>> GetClockStats() {
#if BAND_IS_MOBILE
  std::vector<std::pair<size_t, size_t>> frequency_stats;

  absl::StatusOr<std::vector<size_t>> frequencies =
      GetAvailableFrequenciesKhz();
  absl::StatusOr<std::vector<size_t>> clock_stats =
      TryReadSizeTs(GetPaths("gpu_clock_stats"));

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

}  // namespace gpu
}  // namespace band
