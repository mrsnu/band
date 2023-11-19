#ifndef BAND_PROFILER_PROFILER_H_
#define BAND_PROFILER_PROFILER_H_

#include <chrono>
#include <map>
#include <vector>

#include "absl/status/status.h"

namespace band {

class Profiler {
 public:
  virtual ~Profiler() = default;
  virtual void BeginEvent(JobId job_id) = 0;
  virtual void EndEvent(JobId job_id) = 0;
  virtual size_t GetNumEvents() const = 0;
};

}  // namespace band

#endif  // BAND_PROFILER_PROFILER_H_