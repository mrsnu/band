#include "tensorflow/lite/thermal_zone.h"
#include <cerrno>
#include <cassert>
#include <fstream>

#include "tensorflow/lite/profiling/time.h"

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

} // namespace impl
} // namespace tflite