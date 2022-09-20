#ifndef TENSORFLOW_LITE_RESOURCE_MONITOR_H_
#define TENSORFLOW_LITE_RESOURCE_MONITOR_H_

#include <vector>
#include <string>
#include <unordered_map>
#include <cassert>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/config.h"

namespace tflite {
namespace impl {

typedef int32_t cpu_t;
typedef int32_t thermal_t;
typedef int32_t freq_t;
typedef std::string path_t;
typedef std::string worker_id_t;

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

  TfLiteStatus Init(ResourceConfig& config);

  inline std::string GetThermalZonePath(worker_id_t wid) {
    return tz_path_table_[wid];
  }

  inline TfLiteStatus SetThermalZonePath(worker_id_t wid, path_t path) {
    // Check if the given path is valid.
    if (!CheckPathSanity(path)) {
      return kTfLiteError;
    }
    tz_path_table_.emplace(wid, path);
    // Initialize an empty vector
    thermal_table_.emplace(wid, std::vector<ThermalInfo>());
    return kTfLiteOk;
  }

  inline std::string GetFreqPath(worker_id_t wid) {
    return freq_path_table_[wid];
  }

  inline TfLiteStatus SetFreqPath(worker_id_t wid, path_t path) {
    if (!CheckPathSanity(path)) {
      return kTfLiteError;
    }
    freq_path_table_.emplace(wid, path);
    freq_table_.emplace(wid, std::vector<FreqInfo>());
    return kTfLiteOk;
  }

  thermal_t GetTemperature(worker_id_t);
  std::vector<ThermalInfo> GetTemperatureHistory(worker_id_t);
  ThermalInfo GetTemperatureHistory(worker_id_t, int index);

  freq_t GetFrequency(worker_id_t);
  std::vector<FreqInfo> GetFrequencyHistory(worker_id_t);
  FreqInfo GetFrequencyHistory(worker_id_t, int index);
  
  void ClearHistory(worker_id_t);
  void ClearHistoryAll();
  void DumpAllHistory(path_t log_path);

 private:
  bool CheckPathSanity(path_t path);

  std::unordered_map<worker_id_t, path_t> tz_path_table_;
  std::unordered_map<worker_id_t, path_t> freq_path_table_;
  std::unordered_map<worker_id_t, std::vector<ThermalInfo>> thermal_table_;
  std::unordered_map<worker_id_t, std::vector<FreqInfo>> freq_table_;

  path_t log_path_;
};

} // namespace impl
} // namespace tflite

#endif