#include "tensorflow/lite/processors/processor.h"

#include "tensorflow/lite/processors/cpu.h"
#include "tensorflow/lite/processors/gpu.h"
#include "tensorflow/lite/processors/generic.h"
#include "tensorflow/lite/processors/util.h"

namespace tflite {
namespace impl {
namespace processor {
int GetUpdateIntervalMs(TfLiteDeviceFlags flag, CpuSet cpu_set) {
  if (flag == kTfLiteCPU || flag == kTfLiteCPUFallback) {
    // Use longer interval (Down transition for CPU)
    return cpu::GetDownTransitionLatencyMs(cpu_set);
  } else if (flag == kTfLiteGPU) {
    return gpu::GetPollingIntervalMs();
  } else {
    return generic::GetPollingIntervalMs(flag);
  }
}

int GetFrequencyKhz(TfLiteDeviceFlags flag, CpuSet cpu_set) {
  if (flag == kTfLiteCPU || flag == kTfLiteCPUFallback) {
    return cpu::GetFrequencyKhz(cpu_set);
  } else if (flag == kTfLiteGPU) {
    return gpu::GetFrequencyKhz();
  } else {
    return generic::GetFrequencyKhz(flag);
  }
}

int GetMinFrequencyKhz(TfLiteDeviceFlags flag, CpuSet cpu_set) {
  if (flag == kTfLiteCPU || flag == kTfLiteCPUFallback) {
    return cpu::GetTargetMinFrequencyKhz(cpu_set);
  } else if (flag == kTfLiteGPU) {
    return gpu::GetMinFrequencyKhz();
  } else {
    return generic::GetMinFrequencyKhz(flag);
  }
}

int GetMaxFrequencyKhz(TfLiteDeviceFlags flag, CpuSet cpu_set) {
  if (flag == kTfLiteCPU || flag == kTfLiteCPUFallback) {
    return cpu::GetTargetMaxFrequencyKhz(cpu_set);
  } else if (flag == kTfLiteGPU) {
    return gpu::GetMaxFrequencyKhz();
  } else {
    return generic::GetMaxFrequencyKhz(flag);
  }
}

int GetTargetFrequencyKhz(TfLiteDeviceFlags flag, CpuSet cpu_set) {
  if (flag == kTfLiteCPU || flag == kTfLiteCPUFallback) {
    return cpu::GetTargetFrequencyKhz(cpu_set);
  } else if (flag == kTfLiteGPU) {
    return gpu::GetFrequencyKhz();
  } else {
    return generic::GetTargetFrequencyKhz(flag);
  }
}

std::vector<int> GetAvailableFrequenciesKhz(TfLiteDeviceFlags flag, CpuSet cpu_set) {
  if (flag == kTfLiteCPU || flag == kTfLiteCPUFallback) {
    return cpu::GetAvailableFrequenciesKhz(cpu_set);
  } else if (flag == kTfLiteGPU) {
    return gpu::GetAvailableFrequenciesKhz();
  } else {
    return generic::GetAvailableFrequenciesKhz(flag);
  }
}

}  // namespace processor
}  // namespace impl
}  // namespace tflite