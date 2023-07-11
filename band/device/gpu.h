#ifndef BAND_DEVICE_GPU_H_
#define BAND_DEVICE_GPU_H_

#include <stddef.h>
#include <stdio.h>
#include <vector>
#include <limits.h>
#include "band/common.h"

namespace band {
namespace gpu {

int GetMinFrequencyKhz();
int GetMaxFrequencyKhz();
int GetFrequencyKhz();
int GetPollingIntervalMs();
std::vector<int> GetAvailableFrequenciesKhz();
std::vector<std::pair<int, int>> GetClockStats();

} // namespace gpu
} // namespace band

#endif // BAND_DEVICE_GPU_H_
