#include "tensorflow/lite/profiling/time_profiler.h"
#include <iostream>
#include <chrono>

namespace tflite {
namespace profiling {


TimeProfiler::TimeProfiler() {}

uint32_t TimeProfiler::BeginEvent(const char* tag, EventType event_type,
                                   int64_t event_metadata1,
                                   int64_t event_metadata2) {
  // Matching with TFLITE_SCOPED_TAGGED_DEFAULT_PROFILE in Subgraph's Invoke()
  if(tag == "invoke_subgraph") {
    invoke_timeline_vector_.push_back({std::chrono::system_clock::now(), {}});
    // Returns current timeline's index + 1 as an event handle
    // (to use 0 as an invalid handle of uint32)
    return invoke_timeline_vector_.size();
  } else
    return 0;
}

void TimeProfiler::EndEvent(uint32_t event_handle) {
  if(event_handle && (event_handle - 1 < invoke_timeline_vector_.size())) {
    invoke_timeline_vector_[event_handle - 1].second = std::chrono::system_clock::now();
  }
}

void TimeProfiler::ClearRecords() { invoke_timeline_vector_.clear(); }

size_t TimeProfiler::GetNumInvokeTimelines() const {
  return invoke_timeline_vector_.size();
}

} // namespace profiling
} // namespace tflite 