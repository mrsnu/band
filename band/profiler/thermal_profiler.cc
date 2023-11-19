#include "band/profiler/thermal_profiler.h"

#include <cerrno>
#include <fstream>

#include "band/logger.h"

namespace band {

ThermalProfiler::ThermalProfiler(Thermal* thermal) : thermal_(thermal) {
  BAND_LOG_PROD(BAND_LOG_INFO, "ThermalProfiler is created.");
}

void ThermalProfiler::BeginEvent(JobId job_id) {
  ThermalInfo info = {std::chrono::system_clock::now(),
                      thermal_->GetAllThermal()};
  std::lock_guard<std::mutex> lock(mtx_);
  timeline_[job_id] = {info, {}};
}

void ThermalProfiler::EndEvent(JobId job_id) {
  ThermalInfo info = {std::chrono::system_clock::now(),
                      thermal_->GetAllThermal()};
  timeline_[job_id].second = info;
}

size_t ThermalProfiler::GetNumEvents() const { return timeline_.size(); }

}  // namespace band