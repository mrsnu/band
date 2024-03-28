/*
 * Copyright 2023 Seoul National University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef BAND_TOOL_BENCHMARK_PROFILER_H_
#define BAND_TOOL_BENCHMARK_PROFILER_H_

#include <set>

#include "absl/status/status.h"
#include "band/profiler.h"

namespace band {
namespace tool {

class BenchmarkProfiler : public band::Profiler {
 public:
  BenchmarkProfiler() = default;
  ~BenchmarkProfiler() = default;

  void EndEvent(size_t event_handle, absl::Status status);
  bool IsEventCanceled(size_t event_handle) const;
  size_t GetNumCanceledEvents() const;

 private:
  using band::Profiler::EndEvent;
  std::set<size_t> canceled_events_;
};

}  // namespace tool
}  // namespace band

#endif  // BAND_TOOL_BENCHMARK_PROFILER_H_