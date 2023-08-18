#ifndef BAND_PROFILER_LATENCY_PROFILER_H_
#define BAND_PROFILER_LATENCY_PROFILER_H_

#include <fstream>
#include <chrono>
#include <vector>

#include "band/config.h"
#include "band/profiler/profiler.h"
#include "band/logger.h"

namespace band {

using LatencyInfo = std::pair<std::chrono::system_clock::time_point,
                              std::chrono::system_clock::time_point>;

class LatencyProfiler : public Profiler {
 public:
  size_t BeginEvent() override;
  void EndEvent(size_t event_handle) override;
  size_t GetNumEvents() const override;

  LatencyInfo GetInterval(size_t index) const {
    if (timeline_.size() <= index) {
      return {};
    }
    return timeline_[index];
  }

  double GetDuration(size_t index) const {
    if (timeline_.size() <= index) {
      return -1;
    }
    auto interv = GetInterval(index);
    auto start = interv.first;
    auto end = interv.second;
    return std::max<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count(),
        0);
  }

  double GetAverageDuration() const {
    double accumulated_time = 0;
    for (size_t i = 0; i < timeline_.size(); i++) {
      accumulated_time += GetDuration(i);
    }

    if (timeline_.size() == 0) {
      return 0;
    }

    return accumulated_time / timeline_.size();
  }

 private:
  std::vector<LatencyInfo> timeline_;
};

}  // namespace band

#endif  // BAND_PROFILER_LATENCY_PROFILER_H_