#include "band/profiler/profiler.h"

#include "band/logger.h"

namespace band {

size_t Profiler::BeginEvent() {
  timeline_vector_.push_back({std::chrono::system_clock::now(), {}});
  return timeline_vector_.size();
}

void Profiler::EndEvent(size_t event_handle) {
  if (event_handle && (event_handle - 1 < timeline_vector_.size())) {
    timeline_vector_[event_handle - 1].second =
        std::chrono::system_clock::now();
  } else {
    BAND_LOG_INTERNAL(BAND_LOG_ERROR,
                      "Profiler end event with an invalid handle %d",
                      event_handle);
  }
}

size_t Profiler::GetNumEvents() const { return timeline_vector_.size(); }
}  // namespace band