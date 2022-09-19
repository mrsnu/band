#ifndef TENSORFLOW_LITE_THERMAL_ZONE_H_
#define TENSORFLOW_LITE_THERMAL_ZONE_H_

#include <vector>
#include <string>
#include <unordered_map>
#include <cassert>

#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace impl {

typedef int32_t cpu_t;
typedef int32_t thermal_t;
typedef std::string path_t;
typedef std::string thermal_id_t;

// A singleton instance for reading the temperature from sysfs.
// First, you need to set thermal zone paths calling `SetThermalZonePath`.
class ThermalZoneManager {
 public:
  static ThermalZoneManager& instance() {
    static ThermalZoneManager instance;
    return instance;
  }

  inline std::string GetThermalZonePath(thermal_id_t tid) {
    return tz_path_table_[tid];
  }

  inline TfLiteStatus SetThermalZonePath(thermal_id_t tid, path_t path) {
    // Check if the given path is valid.
    if (!CheckPathSanity(path)) {
      return kTfLiteError;
    }
    tz_path_table_.emplace(tid, path);
    // Initialize an empty vector
    thermal_table_.emplace(tid, std::vector<thermal_t>());
    return kTfLiteOk;
  }

  thermal_t GetTemperature(thermal_id_t);
  
  std::vector<thermal_t> GetTemperatureHistory(thermal_id_t);
  thermal_t GetTemperatureHistory(thermal_id_t, int index);
  
  void ClearHistory(thermal_id_t);
  void ClearHistoryAll();

 private:
  bool CheckPathSanity(path_t path);

  std::unordered_map<thermal_id_t, path_t> tz_path_table_;
  std::unordered_map<thermal_id_t, std::vector<thermal_t>> thermal_table_;
};

} // namespace impl
} // namespace tflite

#endif