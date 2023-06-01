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