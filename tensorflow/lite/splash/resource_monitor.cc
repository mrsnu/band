#include "tensorflow/lite/splash/resource_monitor.h"
#include <cerrno>
#include <cassert>
#include <fstream>

#include "tensorflow/lite/profiling/time.h"
#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "libtflite", __VA_ARGS__)
#include <android/log.h>
#endif // defined(__ANDROID__)
namespace tflite {
namespace impl {

TfLiteStatus ResourceMonitor::Init(ResourceConfig& config) {
  LOGI("Init starts: %d", config.tz_path.size());
  InitTables(config.tz_path.size(), config.freq_path.size());
  // Set all temp/freq path
  for (int i = 0; i < config.tz_path.size(); i++) {
    LOGI("tz_path : %s", config.tz_path[i].c_str());
    SetThermalZonePath(i, config.tz_path[i]);
  }
  for (int i = 0; i < config.freq_path.size(); i++) {
    LOGI("freq_path : %s", config.freq_path[i].c_str());
    SetFreqPath(i, config.freq_path[i]);
  }
  for (int i = 0; i < config.threshold.size(); i++) {
    LOGI("threshold value : %d", config.threshold[i]);
    SetThrottlingThreshold(i, config.threshold[i]);
  }
  for (int i = 0; i < config.target_tz_path.size(); i++) {
    LOGI("target_tz_path : %s", config.target_tz_path[i].c_str());
    SetTargetThermalZonePath(i, config.target_tz_path[i]);
  }
  for (int i = 0; i < config.target_threshold.size(); i++) {
    LOGI("target_threshold value : %d", config.target_threshold[i]);
    SetTargetThreshold(i, config.target_threshold[i]);
  }
  LOGI("Init ends");
  monitor_thread_ = std::thread([this] { this->Monitor(); });
  cpu_set_ = impl::TfLiteCPUMaskGetSet(kTfLiteLittle);
  need_cpu_update_ = true;
  return kTfLiteOk;
}

void ResourceMonitor::Monitor() {
  while (true) {
    if (need_cpu_update_) {
      if (SetCPUThreadAffinity(cpu_set_) != kTfLiteOk) {
        LOGI("[ResourceMonitor] Failed to set cpu thread affinity");
      } 
      need_cpu_update_ = false;
    }
    std::unique_lock<std::mutex> lock(cpu_mtx_);
    // Update temp
    std::vector<thermal_t> ret(kTfLiteNumDevices);
    for (int i = 0; i < temp_table_.size(); i++) {
      temp_table_[i] = ParseTemperature(i);
    }
    // Update frequency
    for (int i = 0; i < freq_table_.size(); i++) {
      freq_table_[i] = ParseFrequency(i);
    }
    // Update target temp
    for (int i = 0; i < target_temp_table_.size(); i++) {
      target_temp_table_[i] = ParseTargetTemperature(i);
    }
    lock.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

bool ResourceMonitor::CheckPathSanity(std::string path) {
  std::ifstream fin;
  fin.open(path);
  if (!fin.is_open()) {
    LOGI("File did not open.\n");
    return false;
  }
  fin.close();
  return true;
}

thermal_t ResourceMonitor::ParseTemperature(worker_id_t wid) {
  std::ifstream fin;
  thermal_t temperature_curr = -1;
  fin.open(GetThermalZonePath(wid));
  if (fin.is_open()) {
    fin >> temperature_curr;
  }
  return temperature_curr;
}

thermal_t ResourceMonitor::ParseTargetTemperature(worker_id_t wid) {
  std::ifstream fin;
  thermal_t temperature_curr = -1;
  fin.open(GetTargetThermalZonePath(wid));
  if (fin.is_open()) {
    fin >> temperature_curr;
  }
  return temperature_curr;
}

freq_t ResourceMonitor::ParseFrequency(worker_id_t wid) {
  std::ifstream fin;
  freq_t freq_curr = -1;
  fin.open(GetFreqPath(wid));
  if (fin.is_open()) {
    fin >> freq_curr;
  }
  return freq_curr;
}


} // namespace impl
} // namespace tflite