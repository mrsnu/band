#ifndef BAND_DEVICE_THERMAL_H_
#define BAND_DEVICE_THERMAL_H_

#include <map>

#include "absl/status/status.h"

namespace band {

class DeviceConfig;
enum class DeviceFlag : size_t;

static const char* GetThermalBasePath() { return "/sys/class/thermal"; }

class Thermal {
 public:
  explicit Thermal(DeviceConfig config);
  size_t GetThermal(DeviceFlag device_flag);
 private:
  bool CheckThermalZone(size_t index);
  std::map<DeviceFlag, size_t> thermal_device_map_;
};

}  // namespace band

#endif  // BAND_DEVICE_THERMAL_H_