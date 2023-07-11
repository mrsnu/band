#include "band/device/device.h"

#include "band/device/cpu.h"
#include "band/device/generic.h"
#include "band/device/gpu.h"
#include "band/device/util.h"

namespace band {
namespace device {
absl::StatusOr<size_t> GetUpdateIntervalMs(DeviceFlag flag, CpuSet cpu_set) {
  if (flag == DeviceFlag::kCPU) {
    // Use longer interval (Down transition for CPU)
    return cpu::GetDownTransitionLatencyMs(cpu_set);
  } else if (flag == DeviceFlag::kGPU) {
    return gpu::GetPollingIntervalMs();
  } else {
    return generic::GetPollingIntervalMs(flag);
  }
}

absl::StatusOr<size_t> GetFrequencyKhz(DeviceFlag flag, CpuSet cpu_set) {
  if (flag == DeviceFlag::kCPU) {
    return cpu::GetFrequencyKhz(cpu_set);
  } else if (flag == DeviceFlag::kGPU) {
    return gpu::GetFrequencyKhz();
  } else {
    return generic::GetFrequencyKhz(flag);
  }
}

absl::StatusOr<size_t> GetMinFrequencyKhz(DeviceFlag flag, CpuSet cpu_set) {
  if (flag == DeviceFlag::kCPU) {
    return cpu::GetTargetMinFrequencyKhz(cpu_set);
  } else if (flag == DeviceFlag::kGPU) {
    return gpu::GetMinFrequencyKhz();
  } else {
    return generic::GetMinFrequencyKhz(flag);
  }
}

absl::StatusOr<size_t> GetMaxFrequencyKhz(DeviceFlag flag, CpuSet cpu_set) {
  if (flag == DeviceFlag::kCPU) {
    return cpu::GetTargetMaxFrequencyKhz(cpu_set);
  } else if (flag == DeviceFlag::kGPU) {
    return gpu::GetMaxFrequencyKhz();
  } else {
    return generic::GetMaxFrequencyKhz(flag);
  }
}

absl::StatusOr<size_t> GetTargetFrequencyKhz(DeviceFlag flag, CpuSet cpu_set) {
  if (flag == DeviceFlag::kCPU) {
    return cpu::GetTargetFrequencyKhz(cpu_set);
  } else if (flag == DeviceFlag::kGPU) {
    return gpu::GetFrequencyKhz();
  } else {
    return generic::GetTargetFrequencyKhz(flag);
  }
}

absl::StatusOr<std::vector<size_t>> GetAvailableFrequenciesKhz(DeviceFlag flag,
                                                               CpuSet cpu_set) {
  if (flag == DeviceFlag::kCPU) {
    return cpu::GetAvailableFrequenciesKhz(cpu_set);
  } else if (flag == DeviceFlag::kGPU) {
    return gpu::GetAvailableFrequenciesKhz();
  } else {
    return generic::GetAvailableFrequenciesKhz(flag);
  }
}

}  // namespace device
}  // namespace band