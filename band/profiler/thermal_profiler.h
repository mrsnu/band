#ifndef BAND_PROFILER_THERMAL_PROFILER_H_
#define BAND_PROFILER_THERMAL_PROFILER_H_

#include <chrono>
#include <vector>

#include "band/config.h"
#include "band/device/thermal.h"
#include "band/profiler/profiler.h"

namespace band {

class ThermalProfiler : public Profiler {
 public:
  explicit ThermalProfiler(DeviceConfig config)
      : thermal_(new Thermal(config)) {}

  size_t BeginEvent() override;
  void EndEvent(size_t event_handle) override;
  size_t GetNumEvents() const override;

  ThermalInfo GetThermalInfoStart(size_t index) const {
    if (thermal_timeline_.size() <= index) {
      return {};
    }
    return thermal_timeline_[index].first;
  }

  ThermalInfo GetThermalInfoEnd(size_t index) const {
    if (thermal_timeline_.size() <= index) {
      return {};
    }
    return thermal_timeline_[index].second;
  }

 private:
  std::unique_ptr<Thermal> thermal_;
  std::vector<std::pair<std::chrono::system_clock::time_point,
                        std::chrono::system_clock::time_point>>
      timeline_;
  std::vector<std::pair<ThermalInfo, ThermalInfo>> thermal_timeline_;
};

}  // namespace band

#endif  // BAND_PROFILER_THERMAL_PROFILER_H_