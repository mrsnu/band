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

#include "band/profiler.h"

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
    BAND_LOG(LogSeverity::kError,
                      "Profiler end event with an invalid handle %d",
                      event_handle);
  }
}

size_t Profiler::GetNumEvents() const { return timeline_vector_.size(); }
}  // namespace band