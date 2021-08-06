#include "tensorflow/lite/processors/processor.h"

#include "tensorflow/lite/processors/cpu.h"
#include "tensorflow/lite/processors/gpu.h"
#include "tensorflow/lite/processors/util.h"

namespace tflite {
namespace impl {
namespace processor {

int GetUpdateIntervalMs(TfLiteDeviceFlags flag, CpuSet cpu_set) {
  if (flag == kTfLiteCPU || flag == kTfLiteCPUFallback) {
    return GetCPUDownTransitionLatencyMs(cpu_set);
  } else if (flag == kTfLiteGPU) {
    return GetGPUPollingIntervalMs();
  }
  return -1;
}

int GetScalingFrequencyKhz(TfLiteDeviceFlags flag, CpuSet cpu_set) {
  if (flag == kTfLiteCPU || flag == kTfLiteCPUFallback) {
    return GetCPUScalingFrequencyKhz(cpu_set);
  } else if (flag == kTfLiteGPU) {
    return GetGPUFrequencyKhz();
  }
  return -1;
}

std::vector<int> GetAvailableFrequenciesKhz(TfLiteDeviceFlags flag, CpuSet cpu_set) {
  if (flag == kTfLiteCPU || flag == kTfLiteCPUFallback) {
    return GetCPUAvailableFrequenciesKhz(cpu_set);
  } else if (flag == kTfLiteGPU) {
    return GetGPUAvailableFrequenciesKhz();
  }
  return {};
}

}  // namespace processor
}  // namespace impl
}  // namespace tflite