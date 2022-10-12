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
  // Set all temp/freq path
  for (int i = 0; i < config.tz_path.size(); i++) {
    LOGI("tz_path : %s", config.tz_path[i].c_str());
    SetThermalZonePath(i, config.tz_path[i]);
  }
  for (int i = 0; i < config.freq_path.size(); i++) {
    LOGI("freq_path : %s", config.freq_path[i].c_str());
    SetFreqPath(i, config.freq_path[i]);
  }
  LOGI("Init ends");
  return kTfLiteOk;
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

std::vector<thermal_t> ResourceMonitor::GetAllTemperature() {
  std::vector<thermal_t> ret(kTfLiteNumDevices);
  for (int i = 0; i < kTfLiteNumDevices; i++) {
    ret[i] = GetTemperature(i);
  }
  return ret;
}

thermal_t ResourceMonitor::GetTemperature(worker_id_t wid) {
  // Ensure that the path is sanitized.
  std::ifstream fin;
  fin.open(GetThermalZonePath(wid));
  assert(fin.is_open());
  thermal_t temperature_curr = -1;
  fin >> temperature_curr;
  return temperature_curr;
}

thermal_t ResourceMonitor::GetThrottlingThreshold(worker_id_t wid) {
  // TODO(chang-jin): calculate threshold from sysfs
  return 80000;
}

std::vector<freq_t> ResourceMonitor::GetAllFrequency() {
  std::vector<freq_t> ret(kTfLiteNumDevices);
  for (int i = 0; i < kTfLiteNumDevices; i++) {
    ret[i] = GetFrequency(i);
  }
  return ret;
}

freq_t ResourceMonitor::GetFrequency(worker_id_t wid) {
  std::ifstream fin;
  fin.open(GetFreqPath(wid));
  assert(fin.is_open());
  freq_t freq_curr = -1;
  fin >> freq_curr;
  return freq_curr;
}

} // namespace impl
} // namespace tflite