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
  ThermalInfo info = {std::chrono::system_clock::now(),
                      thermal_->GetAllThermal()};
  timeline_.push_back({info, {}});
  return timeline_.size() - 1;
}

void ThermalProfiler::EndEvent(size_t event_handle) {
  if (event_handle >= timeline_.size()) {
    BAND_LOG_PROD(BAND_LOG_ERROR,
                  "ThermalProfiler end event with an invalid handle %d",
                  event_handle);
    return;
  }
  ThermalInfo info = {std::chrono::system_clock::now(),
                                        thermal_->GetAllThermal()};
  timeline_[event_handle].second = info;
}

size_t ThermalProfiler::GetNumEvents() const { return timeline_.size(); }

}  // namespace band