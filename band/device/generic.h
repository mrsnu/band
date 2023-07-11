#ifndef BAND_DEVICE_GENERIC_H_
#define BAND_DEVICE_GENERIC_H_

#include <stddef.h>
#include <stdio.h>
#include <vector>
#include <limits.h>
#include "band/common.h"

namespace band {
namespace generic {

// Common helper function for non-CPU generic processors
// that utilizes devfreq structure
// https://www.kernel.org/doc/html/latest/driver-api/devfreq.html

int GetMinFrequencyKhz(DeviceFlag device_flag);
int GetMaxFrequencyKhz(DeviceFlag device_flag);
int GetFrequencyKhz(DeviceFlag device_flag);
int GetTargetFrequencyKhz(DeviceFlag device_flag);
int GetPollingIntervalMs(DeviceFlag device_flag);
std::vector<int> GetAvailableFrequenciesKhz(DeviceFlag device_flag);
std::vector<std::pair<int, int>> GetClockStats(DeviceFlag device_flag);

} // namespace generic
} // namespace band

#endif // BAND_DEVICE_GENERIC_H_
