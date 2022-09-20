#ifndef TENSORFLOW_LITE_RESOURCE_MONITOR_H_
#define TENSORFLOW_LITE_RESOURCE_MONITOR_H_

#include <vector>
#include <string>
#include <unordered_map>
#include <cassert>

#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace impl {

typedef int32_t cpu_t;
typedef int32_t thermal_t;
typedef int32_t freq_t;
typedef std::string path_t;
typedef std::string thermal_id_t;

// Thermal info consists of current time and current temperature.
struct ThermalInfo {
  thermal_t temperature;
  uint64_t time;
};

struct FreqInfo {
  freq_t frequency;
  uint64_t time;
};

// A singleton instance for reading the temperature and frequency from sysfs.
// First, you need to set thermal zone paths calling `SetThermalZonePath`.
class ResourceMonitor {
 public:
  static ResourceMonitor& instance() {
    static ResourceMonitor instance;
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
    thermal_table_.emplace(tid, std::vector<ThermalInfo>());
    return kTfLiteOk;
  }

  thermal_t GetTemperature(thermal_id_t);
  
  std::vector<ThermalInfo> GetTemperatureHistory(thermal_id_t);
  ThermalInfo GetTemperatureHistory(thermal_id_t, int index);
  
  void ClearHistory(thermal_id_t);
  void ClearHistoryAll();

  void SetLogPath(path_t log_path);
  void LogAllHistory();

 private:
  bool CheckPathSanity(path_t path);

  std::unordered_map<thermal_id_t, path_t> tz_path_table_;
  std::unordered_map<thermal_id_t, std::vector<ThermalInfo>> thermal_table_;

  path_t log_path_;
};

} // namespace impl
} // namespace tflite

#endif