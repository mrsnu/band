#ifndef BAND_PROFILER_THERMAL_PROFILER_H_
#define BAND_PROFILER_THERMAL_PROFILER_H_

#include <fstream>
#include <chrono>
#include <vector>

#include "band/config.h"
#include "band/device/thermal.h"
#include "band/profiler/profiler.h"

namespace band {

using ThermalInfo =
    std::pair<std::chrono::system_clock::time_point, ThermalMap>;
using ThermalInterval = std::pair<ThermalInfo, ThermalInfo>;

class ThermalProfiler : public Profiler {
 public:
  explicit ThermalProfiler(DeviceConfig config);
  ~ThermalProfiler();

  size_t BeginEvent() override;
  void EndEvent(size_t event_handle) override;
  size_t GetNumEvents() const override;

  ThermalInterval GetInterval(size_t index) const {
    if (timeline_.size() <= index) {
      return {};
    }
    return timeline_[index];
  }

  ThermalInfo GetStart(size_t index) const {
    return GetInterval(index).first;
  }

  ThermalInfo GetEnd(size_t index) const {
    return GetInterval(index).second;
  }

  ThermalMap GetAllThermal() const {
    return thermal_->GetAllThermal();
  }

 private:
  std::unique_ptr<Thermal> thermal_;
  std::vector<std::pair<ThermalInfo, ThermalInfo>> timeline_;
  std::ofstream log_file_;
};

}  // namespace band

#endif  // BAND_PROFILER_THERMAL_PROFILER_H_