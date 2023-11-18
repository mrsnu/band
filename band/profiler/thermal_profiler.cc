#include "band/profiler/thermal_profiler.h"

#include <fstream>
#include <cerrno>

#include "band/logger.h"

namespace band {

ThermalProfiler::ThermalProfiler(DeviceConfig config)
    : thermal_(new Thermal(config)) {
  BAND_LOG_PROD(BAND_LOG_INFO, "ThermalProfiler is created.");
}

size_t ThermalProfiler::BeginEvent() {
  std::lock_guard<std::mutex> lock(mtx_);
  ThermalInfo info = {std::chrono::system_clock::now(),
                      thermal_->GetAllThermal()};
  timeline_[count_] = {info, {}};
  return count_++;
}

void ThermalProfiler::EndEvent(size_t event_handle) {
  ThermalInfo info = {std::chrono::system_clock::now(),
                                        thermal_->GetAllThermal()};
  timeline_[event_handle].second = info;
}

size_t ThermalProfiler::GetNumEvents() const { return timeline_.size(); }

}  // namespace band