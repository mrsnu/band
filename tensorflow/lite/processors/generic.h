#ifndef TENSORFLOW_LITE_PROCESSORS_GENERIC_H_
#define TENSORFLOW_LITE_PROCESSORS_GENERIC_H_

#include <stddef.h>
#include <stdio.h>
#include <vector>
#include <limits.h>
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace impl {
namespace generic {

// Common helper function for non-CPU generic processors
// that utilizes devfreq structure
// https://www.kernel.org/doc/html/latest/driver-api/devfreq.html

int GetMinFrequencyKhz(TfLiteDeviceFlags device_flag);
int GetMaxFrequencyKhz(TfLiteDeviceFlags device_flag);
int GetFrequencyKhz(TfLiteDeviceFlags device_flag);
int GetTargetFrequencyKhz(TfLiteDeviceFlags device_flag);
int GetPollingIntervalMs(TfLiteDeviceFlags device_flag);
std::vector<int> GetAvailableFrequenciesKhz(TfLiteDeviceFlags device_flag);
std::vector<std::pair<int, int>> GetClockStats(TfLiteDeviceFlags device_flag);

} // namespace generic
} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_PROCESSORS_GENERIC_H_
