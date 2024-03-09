#ifndef BAND_DEVICE_THERMAL_H_
#define BAND_DEVICE_THERMAL_H_

#include <map>

#include "absl/status/status.h"

namespace band {

class DeviceConfig;
enum class SensorFlag : size_t;

using ThermalMap = std::map<SensorFlag, double>;

class Thermal {
 public:
  explicit Thermal(DeviceConfig config);
  double GetThermal(SensorFlag device_flag);
  ThermalMap GetAllThermal();

 private:
  bool CheckThermalZone(size_t index);
  std::map<SensorFlag, size_t> thermal_device_map_;
};

}  // namespace band

#endif  // BAND_DEVICE_THERMAL_H_