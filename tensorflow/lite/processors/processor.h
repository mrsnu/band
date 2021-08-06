#ifndef TENSORFLOW_LITE_PROCESSORS_PROCESSOR_H_
#define TENSORFLOW_LITE_PROCESSORS_PROCESSOR_H_

#include <vector>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/processors/cpu.h"

namespace tflite {
namespace impl {
namespace processor {
int GetUpdateIntervalMs(TfLiteDeviceFlags flag, CpuSet cpu_set = {});
int GetScalingFrequencyKhz(TfLiteDeviceFlags flag, CpuSet cpu_set = {});
std::vector<int> GetAvailableFrequenciesKhz(TfLiteDeviceFlags flag, CpuSet cpu_set = {});
}
}  // namespace impl
}  // namespace tflite

#endif // TENSORFLOW_LITE_PROCESSORS_PROCESSOR_H_