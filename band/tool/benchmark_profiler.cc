// Copyright 2023 Seoul National University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
    BAND_LOG(LogSeverity::kError, "Event %zu failed: %s", event_handle,
                  std::string(status.message()).c_str());
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
