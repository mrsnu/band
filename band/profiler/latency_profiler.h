#ifndef BAND_PROFILER_LATENCY_PROFILER_H_
#define BAND_PROFILER_LATENCY_PROFILER_H_

#include <chrono>
#include <fstream>
#include <vector>

#include "band/config.h"
#include "band/logger.h"
#include "band/profiler/profiler.h"

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

  template <typename T>
  double GetDuration(size_t index) const {
    if (timeline_.size() <= index) {
      BAND_LOG_PROD(BAND_LOG_ERROR, "Index out of bound: %d", index);
      return 0;
    }
    auto interv = GetInterval(index);
    double duration =
        std::chrono::duration_cast<T>(interv.second - interv.first).count();
    return std::max<double>(duration, 0);
  }

  template <typename T>
  double GetAverageDuration() const {
    double accumulated_time = 0;
    for (size_t i = 0; i < timeline_.size(); i++) {
      accumulated_time += GetDuration<T>(i);
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