#ifndef TENSORFLOW_LITE_PROCESSORS_GPU_H_
#define TENSORFLOW_LITE_PROCESSORS_GPU_H_

#include <stddef.h>
#include <stdio.h>
#include <vector>
#include <limits.h>
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace impl {

int GetGPUMinFrequencyKhz();
int GetGPUMaxFrequencyKhz();
int GetGPUFrequencyKhz();
std::vector<int> GetGPUAvailableFrequenciesKhz();
std::vector<std::pair<int, int>> GetGPUClockStats();

} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_PROCESSORS_GPU_H_
