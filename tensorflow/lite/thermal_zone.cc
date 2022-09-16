#include "tensorflow/lite/thermal_zone.h"

#include <cerrno>
#include <cassert>

#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace impl {

bool ThermalZoneManager::CheckPathSanity(std::string path) {
  FILE* fp = fopen(path.c_str(), "r");
  if (fp == nullptr) {
    TFLITE_LOG_PROD(TFLITE_LOG_INFO, "File open failed: %d", errno);
    return false;
  }

  fclose(fp);
  return true;
}

thermal_t ThermalZoneManager::GetTemperature(thermal_id_t tid) {
    FILE* fp = fopen(GetThermalZonePath(tid).c_str(), "r");
    // Ensure that the path is sanitized.
    assert(fp != nullptr);
    thermal_t temperature_curr;
    fscanf(fp, "%d", &temperature_curr);
    // TODO(widiba03304): figure out how to find availability.
    if (temperature_curr < 0) {
      // Negative value indicates the disabled status.
      return -1;
    }
    thermal_table_[tid].push_back(temperature_curr);
    return temperature_curr;
  }

} // namespace impl
} // namespace tflite