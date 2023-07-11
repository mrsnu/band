#ifndef BAND_DEVICE_DEVICE_H_
#define BAND_DEVICE_DEVICE_H_

#include <vector>

#include "band/common.h"
#include "band/device/cpu.h"

namespace band {
namespace device {
absl::StatusOr<size_t> GetUpdateIntervalMs(DeviceFlag flag,
                                           CpuSet cpu_set = {});
absl::StatusOr<size_t> GetFrequencyKhz(DeviceFlag flag, CpuSet cpu_set = {});
absl::StatusOr<size_t> GetMinFrequencyKhz(DeviceFlag flag, CpuSet cpu_set = {});
absl::StatusOr<size_t> GetMaxFrequencyKhz(DeviceFlag flag, CpuSet cpu_set = {});
absl::StatusOr<size_t> GetTargetFrequencyKhz(DeviceFlag flag,
                                             CpuSet cpu_set = {});
absl::StatusOr<std::vector<size_t>> GetAvailableFrequenciesKhz(
    DeviceFlag flag, CpuSet cpu_set = {});
}  // namespace device
}  // namespace band

#endif  // BAND_DEVICE_DEVICE_H_