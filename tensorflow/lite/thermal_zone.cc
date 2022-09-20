#include "tensorflow/lite/thermal_zone.h"
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

bool ThermalZoneManager::CheckPathSanity(std::string path) {
  std::ifstream fin;
  fin.open(path);
  if (!fin.is_open()) {
    printf("File did not open.\n");
    return false;
  }
  fin.close();
  return true;
}

thermal_t ThermalZoneManager::GetTemperature(thermal_id_t tid) {
  //int fp = fopen(GetThermalZonePath(tid).c_str(), O_RDONLY);
  // Ensure that the path is sanitized.
  std::ifstream fin;
  fin.open(GetThermalZonePath(tid));
  assert(fin.is_open());
  thermal_t temperature_curr = -1;
  auto time_curr = profiling::time::NowMicros();
  fin >> temperature_curr;
  // TODO(widiba03304): figure out how to find availability.
  if (temperature_curr < 0) {
    // Negative value indicates the disabled status.
    return -1;
  }
  ThermalInfo info_curr = { time_curr, temperature_curr };
  thermal_table_[tid].push_back(info_curr);
  return temperature_curr;
}

std::vector<ThermalInfo> ThermalZoneManager::GetTemperatureHistory(thermal_id_t tid) {
  return thermal_table_[tid];
}

ThermalInfo ThermalZoneManager::GetTemperatureHistory(thermal_id_t tid, int index) {
  return thermal_table_[tid][index];
}

void ThermalZoneManager::ClearHistory(thermal_id_t tid) {
  thermal_table_[tid].clear();
}

void ThermalZoneManager::ClearHistoryAll() {
  for (auto& history : thermal_table_) {
    history.second.clear();
  }
}

void ThermalZoneManager::SetLogPath(path_t log_path) {
  log_path_ = log_path;
  if (log_path_.size()) {
    std::ofstream log_file(log_path_);
    if (!log_file.is_open()) return kTfLiteError;
    log_file << "current_time\t"
             << "current_temperature\n";
    log_file.close();
  } else {
    LOGI("[ThermalManager] Invalid log file path %s", log_path_.c_str());
  }
}

void ThermalZoneManager::LogAllHistory() {
  std::ofstream log_file(log_path_, std::ofstream::app);
  if (log_file.is_open()) {
    for (auto& history : thermal_table_) {
      thermal_id_t t_id = history.first;
      for (auto t_info : history.second) {
        log_file << t_info.time << "\t"
                << t_info.temperature << "\n";
      }
    } 
    log_file.close();
  }
}

} // namespace impl
} // namespace tflite