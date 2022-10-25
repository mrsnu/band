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

typedef int32_t worker_id_t;
typedef int32_t thermal_t;
typedef int32_t freq_t;
typedef std::string path_t;

// A singleton instance for reading the temperature and frequency from sysfs.
// First, you need to set thermal zone paths calling `SetThermalZonePath`.
class ResourceMonitor {
 public:
  static ResourceMonitor& instance() {
    static ResourceMonitor instance;
    return instance;
  }

  TfLiteStatus Init(ResourceConfig& config);

  inline void InitTables(int tz_size, int freq_size) {
    for (int i = 0; i < tz_size ; i++) {
      tz_path_table_.push_back("");
      throttling_threshold_table_.push_back(INT_MAX);
    }
    for (int i = 0; i < freq_size ; i++) {
      freq_path_table_.push_back("");
    }
  }

  inline std::string GetThermalZonePath(worker_id_t wid) {
    return tz_path_table_[wid];
  }

  inline TfLiteStatus SetThermalZonePath(worker_id_t wid, path_t path) {
    if (!CheckPathSanity(path)) {
      return kTfLiteError;
    }
    tz_path_table_[wid] = path;
    return kTfLiteOk;
  }

  inline std::string GetFreqPath(worker_id_t wid) {
    return freq_path_table_[wid];
  }

  inline TfLiteStatus SetFreqPath(worker_id_t wid, path_t path) {
    if (!CheckPathSanity(path)) {
      return kTfLiteError;
    }
    freq_path_table_[wid] = path;
    return kTfLiteOk;
  }

  inline thermal_t GetThrottlingThreshold(worker_id_t wid) {
    return throttling_threshold_table_[wid];
  }

  inline TfLiteStatus SetThrottlingThreshold(worker_id_t wid, thermal_t value) {
    throttling_threshold_table_[wid] = value;
    return kTfLiteOk;
  }

  std::vector<thermal_t> GetAllTemperature();
  std::vector<freq_t> GetAllFrequency();

  thermal_t GetTemperature(worker_id_t wid);
  freq_t GetFrequency(worker_id_t wid);

 private:
  bool CheckPathSanity(path_t path);

  std::vector<path_t> tz_path_table_;
  std::vector<path_t> freq_path_table_;
  std::vector<thermal_t> throttling_threshold_table_;
};

} // namespace impl
} // namespace tflite

#endif