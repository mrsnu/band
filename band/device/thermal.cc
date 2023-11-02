#include "band/device/thermal.h"

#include "absl/strings/str_format.h"
#include "band/common.h"
#include "band/config.h"
#include "band/device/util.h"
#include "band/logger.h"

namespace band {

namespace {

const char* GetThermalBasePath() { return "/sys/class/thermal"; }

std::string GetThermalPath(size_t index) {
  return absl::StrFormat("%s/thermal_zone%d/temp", GetThermalBasePath(), index);
}

}  // anonymous namespace

Thermal::Thermal(DeviceConfig config) {
  device::Root();
  
  if (config.cpu_therm_index != -1 &&
      CheckThermalZone(config.cpu_therm_index)) {
    thermal_device_map_[SensorFlag::kCPU] = config.cpu_therm_index;
  } else {
    BAND_LOG_PROD(BAND_LOG_ERROR, "CPU thermal zone %d is not available.",
                  config.cpu_therm_index);
  }

  if (config.gpu_therm_index != -1 &&
      CheckThermalZone(config.gpu_therm_index)) {
    thermal_device_map_[SensorFlag::kGPU] = config.gpu_therm_index;
  } else {
    BAND_LOG_PROD(BAND_LOG_ERROR, "GPU thermal zone %d is not available.",
                  config.gpu_therm_index);
  }

  if (config.dsp_therm_index != -1 &&
      CheckThermalZone(config.dsp_therm_index)) {
    thermal_device_map_[SensorFlag::kDSP] = config.dsp_therm_index;
  } else {
    BAND_LOG_PROD(BAND_LOG_ERROR, "DSP thermal zone %d is not available.",
                  config.dsp_therm_index);
  }

  if (config.npu_therm_index != -1 &&
      CheckThermalZone(config.npu_therm_index)) {
    thermal_device_map_[SensorFlag::kNPU] = config.npu_therm_index;
  } else {
    BAND_LOG_PROD(BAND_LOG_ERROR, "NPU thermal zone %d is not available.",
                  config.npu_therm_index);
  }

  if (config.target_therm_index != -1 &&
      CheckThermalZone(config.target_therm_index)) {
    thermal_device_map_[SensorFlag::kTarget] = config.target_therm_index;
  } else {
    BAND_LOG_PROD(BAND_LOG_ERROR, "Target thermal zone %d is not available.",
                  config.target_therm_index);
  }
}

double Thermal::GetThermal(SensorFlag device_flag) {
  auto path = GetThermalPath(thermal_device_map_[device_flag]);
  return device::TryReadDouble({path}, {0.001}).value();
}

ThermalMap Thermal::GetAllThermal() {
  ThermalMap thermal_map;
  for (auto& pair : thermal_device_map_) {
    thermal_map[pair.first] = GetThermal(pair.first);
  }
  return thermal_map;
}

bool Thermal::CheckThermalZone(size_t index) {
  return device::IsFileAvailable(GetThermalPath(index));
}

}  // namespace band