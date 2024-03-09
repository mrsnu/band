#ifndef BAND_PROFILER_THERMAL_PROFILER_H_
#define BAND_PROFILER_THERMAL_PROFILER_H_

#include <fstream>
#include <chrono>
#include <vector>
#include <map>
#include <mutex>

#include "band/common.h"
#include "band/device/thermal.h"
#include "band/profiler/profiler.h"

namespace band {

using ThermalInfo =
    std::pair<std::chrono::system_clock::time_point, ThermalMap>;
using ThermalInterval = std::pair<ThermalInfo, ThermalInfo>;

class ThermalProfiler : public Profiler {
 public:
  explicit ThermalProfiler(Thermal* thermal);
  ~ThermalProfiler() {}

  void BeginEvent(JobId job_id) override;
  void EndEvent(JobId job_id) override;
  size_t GetNumEvents() const override;

  ThermalInterval GetInterval(size_t index) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (timeline_.find(index) == timeline_.end()) {
      return {{}, {}};
    }
    ThermalInterval result = timeline_.at(index);
    timeline_.erase(index);
    return result;
  }

  ThermalMap GetAllThermal() const {
    return thermal_->GetAllThermal();
  }

  Thermal* GetThermal() const {
    return thermal_;
  }

 private:
  size_t count_ = 0;

  Thermal* thermal_;
  std::map<JobId, ThermalInterval> timeline_;

  mutable std::mutex mtx_;
};

}  // namespace band

#endif  // BAND_PROFILER_THERMAL_PROFILER_H_