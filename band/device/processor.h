#ifndef BAND_DEVICE_PROCESSOR_H_
#define BAND_DEVICE_PROCESSOR_H_

#include <vector>
#include "band/common.h"
#include "band/device/cpu.h"

namespace band {
namespace processor {
int GetUpdateIntervalMs(DeviceFlag flag, CpuSet cpu_set = {});
int GetFrequencyKhz(DeviceFlag flag, CpuSet cpu_set = {});
int GetMinFrequencyKhz(DeviceFlag flag, CpuSet cpu_set = {});
int GetMaxFrequencyKhz(DeviceFlag flag, CpuSet cpu_set = {});
int GetTargetFrequencyKhz(DeviceFlag flag, CpuSet cpu_set = {});
std::vector<int> GetAvailableFrequenciesKhz(DeviceFlag flag, CpuSet cpu_set = {});
}
}  // namespace band

#endif // BAND_DEVICE_PROCESSOR_H_