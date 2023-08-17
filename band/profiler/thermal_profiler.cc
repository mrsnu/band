#include "band/profiler/thermal_profiler.h"

#include "band/logger.h"

namespace band {

size_t ThermalProfiler::BeginEvent() {
  timeline_.push_back({std::chrono::system_clock::now(), {}});
  thermal_timeline_.push_back({thermal_->GetAllThermal(), {}});
  return timeline_.size();
}

void ThermalProfiler::EndEvent(size_t event_handle) {
  if (event_handle && (event_handle - 1 < timeline_.size())) {
    timeline_[event_handle - 1].second =
        std::chrono::system_clock::now();
    thermal_timeline_[event_handle - 1].second = thermal_->GetAllThermal();
  } else {
    BAND_LOG_INTERNAL(BAND_LOG_ERROR,
                      "ThermalProfiler end event with an invalid handle %d",
                      event_handle);
  }
}

size_t ThermalProfiler::GetNumEvents() const { return timeline_.size(); }

}  // namespace band