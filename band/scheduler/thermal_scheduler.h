#ifndef BAND_SCHEDULER_THERMAL_SCHEDULER_H_
#define BAND_SCHEDULER_THERMAL_SCHEDULER_H_

#include "band/scheduler/scheduler.h"

namespace band {

class ThermalScheduler : public IScheduler {
 public:
  explicit ThermalScheduler(IEngine& engine, int window_size, double eta);

  bool Schedule(JobQueue& requests) override;
  bool NeedFallbackSubgraphs() override { return true; }
  WorkerType GetWorkerType() override { return WorkerType::kGlobalQueue; }

 private:
  double eta_ = 0.1f;
  const int window_size_;
};

}  // namespace band

#endif  // BAND_SCHEDULER_THERMAL_SCHEDULER_H_