#ifndef BAND_PROFILER_THERMAL_PROFILER_H_
#define BAND_PROFILER_THERMAL_PROFILER_H_

#include <chrono>
#include <vector>

#include "band/profiler/profiler.h"

namespace band {

class ThermalProfiler : public Profiler {
 public:
  size_t BeginEvent() override;
  void EndEvent(size_t event_handle) override;
  size_t GetNumEvents() const override;

  double GetInterval(size_t index) const override {
    if (timeline_.size() > index) {
      return std::max<double>(
          std::chrono::duration_cast<std::chrono::microseconds>(
              timeline_[index].second - timeline_[index].first)
              .count(),
          0);
    } else
      return 0;
  }

  double GetAverageInterval() const override {
    double accumulated_time = 0;
    for (size_t i = 0; i < timeline_.size(); i++) {
      accumulated_time += GetInterval(i);
    }

    if (timeline_.size() == 0) {
      return 0;
    }

    return accumulated_time / timeline_.size();
  }

 private:
  std::vector<std::pair<std::chrono::system_clock::time_point,
                        std::chrono::system_clock::time_point>>
      timeline_;
};

}  // namespace band

#endif  // BAND_PROFILER_THERMAL_PROFILER_H_