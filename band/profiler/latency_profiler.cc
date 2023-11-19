#include "band/profiler/latency_profiler.h"

#include "band/logger.h"

namespace band {

void LatencyProfiler::BeginEvent(JobId job_id) {
  std::lock_guard<std::mutex> lock(mtx_);
  timeline_[job_id] = {std::chrono::system_clock::now(), {}};
}

void LatencyProfiler::EndEvent(JobId job_id) {
  std::lock_guard<std::mutex> lock(mtx_);
  timeline_[job_id].second = std::chrono::system_clock::now();
}

size_t LatencyProfiler::GetNumEvents() const { return timeline_.size(); }

}  // namespace band