#ifndef BAND_DEVICE_GENERIC_H_
#define BAND_DEVICE_GENERIC_H_

#include <limits.h>
#include <stddef.h>
#include <stdio.h>

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "band/common.h"

namespace band {
namespace generic {

// Common helper function for non-CPU generic processors
// that utilizes devfreq structure
// https://www.kernel.org/doc/html/latest/driver-api/devfreq.html

absl::StatusOr<size_t> GetMinFrequencyKhz(DeviceFlag device_flag);
absl::StatusOr<size_t> GetMaxFrequencyKhz(DeviceFlag device_flag);
absl::StatusOr<size_t> GetFrequencyKhz(DeviceFlag device_flag);
absl::StatusOr<size_t> GetTargetFrequencyKhz(DeviceFlag device_flag);
absl::StatusOr<size_t> GetPollingIntervalMs(DeviceFlag device_flag);
absl::StatusOr<std::vector<size_t>> GetAvailableFrequenciesKhz(
    DeviceFlag device_flag);
absl::StatusOr<std::vector<std::pair<size_t, size_t>>> GetClockStats(
    DeviceFlag device_flag);

}  // namespace generic
}  // namespace band

#endif  // BAND_DEVICE_GENERIC_H_
