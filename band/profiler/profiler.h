#ifndef BAND_PROFILER_PROFILER_H_
#define BAND_PROFILER_PROFILER_H_

#include <chrono>
#include <map>
#include <vector>

namespace band {

class Profiler {
 public:
  virtual ~Profiler() = default;
  virtual size_t BeginEvent() = 0;
  virtual void EndEvent(size_t event_handle) = 0;
  virtual size_t GetNumEvents() const = 0;
};

}  // namespace band

#endif  // BAND_PROFILER_PROFILER_H_