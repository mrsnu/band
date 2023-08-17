#include "band/profiler/frequency_profiler.h"

#include "band/logger.h"

namespace band {

size_t FrequencyProfiler::BeginEvent() {
  timeline_.push_back({std::chrono::system_clock::now(), {}});
  frequency_timeline_.push_back({frequency_->GetAllFrequency(), {}});
  return timeline_.size();
}

void FrequencyProfiler::EndEvent(size_t event_handle) {
  if (event_handle && (event_handle - 1 < timeline_.size())) {
    timeline_[event_handle - 1].second = std::chrono::system_clock::now();
    frequency_timeline_[event_handle - 1].second =
        frequency_->GetAllFrequency();
  } else {
    BAND_LOG_INTERNAL(BAND_LOG_ERROR,
                      "FrequencyProfiler end event with an invalid handle %d",
                      event_handle);
  }
}

size_t FrequencyProfiler::GetNumEvents() const { return timeline_.size(); }

}  // namespace band
