#ifndef TENSORFLOW_LITE_PROCESSORS_GPU_H_
#define TENSORFLOW_LITE_PROCESSORS_GPU_H_

#include <stddef.h>
#include <stdio.h>
#include <vector>
#include <limits.h>
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace impl {
namespace gpu {

int GetMinFrequencyKhz();
int GetMaxFrequencyKhz();
int GetFrequencyKhz();
int GetPollingIntervalMs();
std::vector<int> GetAvailableFrequenciesKhz();
std::vector<std::pair<int, int>> GetClockStats();

} // namespace gpu
} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_PROCESSORS_GPU_H_
