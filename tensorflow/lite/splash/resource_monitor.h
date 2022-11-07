#ifndef TENSORFLOW_LITE_RESOURCE_MONITOR_H_
#define TENSORFLOW_LITE_RESOURCE_MONITOR_H_

#include <vector>
#include <string>
#include <unordered_map>
#include <cassert>
#include <thread>

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

  void Monitor();

  inline void InitTables(int tz_size, int freq_size) {
    for (int i = 0; i < tz_size ; i++) {
      tz_path_table_.push_back("");
      throttling_threshold_table_.push_back(INT_MAX);
      target_tz_path_table_.push_back("");
      target_threshold_table_.push_back(INT_MAX);

      temp_table_.push_back(0);
      target_temp_table_.push_back(0);
    }
    for (int i = 0; i < freq_size ; i++) {
      freq_path_table_.push_back("");
      freq_table_.push_back(0);
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

  inline std::string GetTargetThermalZonePath(worker_id_t wid) {
    return target_tz_path_table_[wid];
  }

  inline TfLiteStatus SetTargetThermalZonePath(worker_id_t wid, path_t path) {
    if (!CheckPathSanity(path)) {
      return kTfLiteError;
    }
    target_tz_path_table_[wid] = path;
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

  inline thermal_t GetTargetThreshold(worker_id_t wid) {
    return target_threshold_table_[wid];
  }

  inline TfLiteStatus SetTargetThreshold(worker_id_t wid, thermal_t value) {
    target_threshold_table_[wid] = value;
    return kTfLiteOk;
  }

  inline std::vector<thermal_t> GetAllTemperature() {
    return temp_table_;
  }

  inline std::vector<thermal_t> GetAllTargetTemperature() {
    return target_temp_table_;
  }

  inline void FillJobInfoBefore(Job& job) {
    job.before_temp = GetAllTemperature();
    job.before_target_temp = GetAllTargetTemperature();
    job.frequency = GetAllFrequency(); 
  }

  inline void FillJobInfoAfter(Job& job) {
    job.after_temp = GetAllTemperature();
    job.after_target_temp = GetAllTargetTemperature();
  }

  inline std::vector<freq_t> GetAllFrequency() {
    return freq_table_;
  }

  inline thermal_t GetTemperature(worker_id_t wid) {
    std::unique_lock<std::mutex> lock(cpu_mtx_);
    return temp_table_[wid];
  }

  inline thermal_t GetTargetTemperature(worker_id_t wid) {
    std::unique_lock<std::mutex> lock(cpu_mtx_);
    return target_temp_table_[wid];
  }

  inline freq_t GetFrequency(worker_id_t wid) {
    std::unique_lock<std::mutex> lock(cpu_mtx_);
    return freq_table_[wid];
  }

 private:
  bool CheckPathSanity(path_t path);

  thermal_t ParseTemperature(worker_id_t wid);
  thermal_t ParseTargetTemperature(worker_id_t wid);
  freq_t ParseFrequency(worker_id_t wid);

  std::thread monitor_thread_;
  CpuSet cpu_set_;
  bool need_cpu_update_ = false;
  std::mutex cpu_mtx_;

  std::vector<path_t> tz_path_table_;
  std::vector<path_t> freq_path_table_;
  std::vector<path_t> target_tz_path_table_;

  std::vector<thermal_t> throttling_threshold_table_;
  std::vector<thermal_t> target_threshold_table_;

  std::vector<thermal_t> temp_table_;
  std::vector<freq_t> freq_table_;
  std::vector<thermal_t> target_temp_table_;
};

} // namespace impl
} // namespace tflite

#endif