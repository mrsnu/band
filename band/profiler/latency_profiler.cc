#include "band/profiler/latency_profiler.h"

#include "band/logger.h"

namespace band {

size_t LatencyProfiler::BeginEvent() {
  timeline_.push_back({std::chrono::system_clock::now(), {}});
  return timeline_.size();
}

void LatencyProfiler::EndEvent(size_t event_handle) {
  if (event_handle && (event_handle - 1 < timeline_.size())) {
    timeline_[event_handle - 1].second =
        std::chrono::system_clock::now();
  } else {
    BAND_LOG_INTERNAL(BAND_LOG_ERROR,
                      "LatencyProfiler end event with an invalid handle %d",
                      event_handle);
  }
}

size_t LatencyProfiler::GetNumEvents() const { return timeline_.size(); }

}  // namespace band