#include "band/device/thermal.h"

#include "absl/strings/str_format.h"
#include "band/common.h"
#include "band/config.h"
#include "band/device/util.h"

namespace band {

namespace {

const char* GetThermalBasePath() { return "/sys/class/thermal"; }

std::string GetThermalPath(size_t index) {
  return absl::StrFormat("%s/thermal_zone%d/temp", GetThermalBasePath(), index);
}

}  // anonymous namespace

Thermal::Thermal(DeviceConfig config) {
  if (config.cpu_therm_index != -1 &&
      !CheckThermalZone(config.cpu_therm_index)) {
    thermal_device_map_[DeviceFlag::kCPU] = config.cpu_therm_index;
  }

  if (config.gpu_therm_index != -1 &&
      !CheckThermalZone(config.gpu_therm_index)) {
    thermal_device_map_[DeviceFlag::kGPU] = config.gpu_therm_index;
  }

  if (config.dsp_therm_index != -1 &&
      !CheckThermalZone(config.dsp_therm_index)) {
    thermal_device_map_[DeviceFlag::kDSP] = config.dsp_therm_index;
  }

  if (config.npu_therm_index != -1 &&
      !CheckThermalZone(config.npu_therm_index)) {
    thermal_device_map_[DeviceFlag::kNPU] = config.npu_therm_index;
  }
}

size_t Thermal::GetThermal(DeviceFlag device_flag) {
  auto path = GetThermalPath(thermal_device_map_[device_flag]);
  return device::TryReadSizeT({path}).value();
}

ThermalMap Thermal::GetAllThermal() {
  std::map<DeviceFlag, size_t> thermal_map;
  for (auto& pair : thermal_device_map_) {
    thermal_map[pair.first] = GetThermal(pair.first);
  }
  return thermal_map;
}

bool Thermal::CheckThermalZone(size_t index) {
  return device::IsFileAvailable(GetThermalPath(index));
}

}  // namespace band