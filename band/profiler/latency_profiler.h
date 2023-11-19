#ifndef BAND_PROFILER_LATENCY_PROFILER_H_
#define BAND_PROFILER_LATENCY_PROFILER_H_

#include <chrono>
#include <fstream>
#include <mutex>
#include <vector>

#include "band/config.h"
#include "band/logger.h"
#include "band/profiler/profiler.h"

namespace band {

using LatencyInterval = std::pair<std::chrono::system_clock::time_point,
                                  std::chrono::system_clock::time_point>;

class LatencyProfiler : public Profiler {
 public:
  void BeginEvent(JobId job_id) override;
  void EndEvent(JobId job_id) override;
  size_t GetNumEvents() const override;

  LatencyInterval GetInterval(JobId job_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (timeline_.find(job_id) == timeline_.end()) {
      return {{}, {}};
    }
    LatencyInterval result = timeline_.at(job_id);
    timeline_.erase(job_id);
    return result;
  }

 private:
  size_t count_ = 0;

  std::map<JobId, LatencyInterval> timeline_;

  mutable std::mutex mtx_;
};

}  // namespace band

#endif  // BAND_PROFILER_LATENCY_PROFILER_H_