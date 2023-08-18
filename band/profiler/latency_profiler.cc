#include "band/profiler/latency_profiler.h"

#include "band/logger.h"

namespace band {

size_t LatencyProfiler::BeginEvent() {
  timeline_.push_back({std::chrono::system_clock::now(), {}});
  return timeline_.size() - 1;
}

void LatencyProfiler::EndEvent(size_t event_handle) {
  if (!event_handle || event_handle >= timeline_.size()) {
    BAND_LOG_PROD(BAND_LOG_ERROR,
                  "Invalid event handle: %lu (timeline size: %lu)",
                  event_handle, timeline_.size());
    return;
  }
  timeline_[event_handle].second = std::chrono::system_clock::now();
}

size_t LatencyProfiler::GetNumEvents() const { return timeline_.size(); }

}  // namespace band