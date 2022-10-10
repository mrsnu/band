#include "tensorflow/lite/splash/resource_monitor.h"
#include <cerrno>
#include <cassert>
#include <fstream>

#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace impl {

TfLiteStatus ResourceMonitor::Init(ResourceConfig& config) {
  log_path_ = config.temperature_log_path;
  if (log_path_.size()) {
    std::ofstream log_file(log_path_);
    if (!log_file.is_open()) return kTfLiteError;
    log_file << "current_time\t"
             << "current_temperature\n";
    log_file.close();
  }

  // Set all temp/freq path
  for (auto tz : config.tz_path) {
    SetThermalZonePath(tz.first, tz.second);
  }
  for (auto freq: config.freq_path) {
    SetFreqPath(freq.first, freq.second);
  }
}

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

thermal_t ResourceMonitor::GetThrottlingThreshold(worker_id_t wid) {
  // TODO(chang-jin): calculate threshold from sysfs
  return 80000;
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
    log_file << "current_time\t"
             << "current_temperature\n";
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