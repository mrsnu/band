#ifndef BAND_DEVICE_GPU_H_
#define BAND_DEVICE_GPU_H_

#include <limits.h>
#include <stddef.h>
#include <stdio.h>

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "band/common.h"

namespace band {
namespace gpu {

absl::StatusOr<size_t> GetMinFrequencyKhz();
absl::StatusOr<size_t> GetMaxFrequencyKhz();
absl::StatusOr<size_t> GetFrequencyKhz();
absl::StatusOr<size_t> GetPollingIntervalMs();
absl::StatusOr<std::vector<size_t>> GetAvailableFrequenciesKhz();
absl::StatusOr<std::vector<std::pair<size_t, size_t>>> GetClockStats();

}  // namespace gpu
}  // namespace band

#endif  // BAND_DEVICE_GPU_H_
