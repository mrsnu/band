#include "band/tool/benchmark_profiler.h"

#include "band/logger.h"

namespace band {
namespace tool {
void BenchmarkProfiler::EndEvent(size_t event_handle, absl::Status status) {
  if (status.ok()) {
    band::Profiler::EndEvent(event_handle);
  } else if (status.code() == absl::StatusCode::kDeadlineExceeded) {
    canceled_events_.insert(event_handle);
  } else {
    BAND_LOG_PROD(BAND_LOG_ERROR, "Event %zu failed: %s", event_handle,
                  status.message());
  }
}

bool BenchmarkProfiler::IsEventCanceled(size_t event_handle) const {
  return canceled_events_.find(event_handle) != canceled_events_.end();
}
size_t BenchmarkProfiler::GetNumCanceledEvents() const {
  return canceled_events_.size();
}
}  // namespace tool
}  // namespace band
