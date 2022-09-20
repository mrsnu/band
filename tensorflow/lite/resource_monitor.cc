#include "tensorflow/lite/resource_monitor.h"
#include <cerrno>
#include <cassert>
#include <fstream>

#include "tensorflow/lite/profiling/time.h"

#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "libtflite", __VA_ARGS__)
#include <android/log.h>
#else
#define LOGI(...) printf(__VA_ARGS__)
#endif // defined(__ANDROID__)

namespace tflite {
namespace impl {

bool ResourceMonitor::CheckPathSanity(std::string path) {
  std::ifstream fin;
  fin.open(path);
  if (!fin.is_open()) {
    printf("File did not open.\n");
    return false;
  }
  fin.close();
  return true;
}

thermal_t ResourceMonitor::GetTemperature(worker_id_t wid) {
  // Ensure that the path is sanitized.
  std::ifstream fin;
  fin.open(GetThermalZonePath(wid));
  assert(fin.is_open());
  thermal_t temperature_curr = -1;
  uint64_t time_curr = profiling::time::NowMicros();
  fin >> temperature_curr;
  // TODO(widiba03304): figure out how to find availability.
  if (temperature_curr < 0) {
    // Negative value indicates the disabled status.
    return -1;
  }
  ThermalInfo info_curr = { temperature_curr, time_curr };
  thermal_table_[wid].push_back(info_curr);
  return temperature_curr;
}

std::vector<ThermalInfo> ResourceMonitor::GetTemperatureHistory(worker_id_t wid) {
  return thermal_table_[wid];
}

ThermalInfo ResourceMonitor::GetTemperatureHistory(worker_id_t wid, int index) {
  return thermal_table_[wid][index];
}

freq_t ResourceMonitor::GetFrequency(worker_id_t wid) {
  std::ifstream fin;
  fin.open(GetFreqPath(wid));
  assert(fin.is_open());
  freq_t freq_curr = -1;
  uint64_t time_curr = profiling::time::NowMicros();
  fin >> freq_curr;
  // TODO(widiba03304): figure out how to find availability.
  if (freq_curr < 0) {
    return -1;
  }
  FreqInfo info_curr = { freq_curr, time_curr };
  freq_table_[wid].push_back(info_curr);
  return freq_curr;
}

std::vector<FreqInfo> ResourceMonitor::GetFrequencyHistory(worker_id_t wid) {
  return freq_table_[wid];
}

FreqInfo ResourceMonitor::GetFrequencyHistory(worker_id_t wid, int index) {
  return freq_table_[wid][index];
}

void ResourceMonitor::ClearHistory(worker_id_t wid) {
  thermal_table_[wid].clear();
}

void ResourceMonitor::ClearHistoryAll() {
  for (auto& history : thermal_table_) {
    history.second.clear();
  }
}

void ResourceMonitor::DumpAllHistory(path_t log_path) {
  std::ofstream log_file(log_path, std::ofstream::app);
  if (log_file.is_open()) {
    for (auto& history : thermal_table_) {
      worker_id_t t_id = history.first;
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