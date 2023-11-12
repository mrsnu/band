#ifndef BAND_SCHEDULER_FIXED_WORKER_IDLE_SCHEDULER_H_
#define BAND_SCHEDULER_FIXED_WORKER_IDLE_SCHEDULER_H_

#include "band/scheduler/scheduler.h"

namespace band {

// Assigns requested model to devices according to a direct request from engine
// or model_id.
class FixedWorkerIdleScheduler : public IScheduler {
 public:
  using IScheduler::IScheduler;
  explicit FixedWorkerIdleScheduler(IEngine& engine, int idle_us)
      : IScheduler(engine), idle_us_(idle_us) {}

  bool Schedule(JobQueue& requests) override;
  bool NeedFallbackSubgraphs() override { return false; }
  WorkerType GetWorkerType() override { return WorkerType::kDeviceQueue; }

 private:
  int idle_us_ = 5000;
};

}  // namespace band

#endif  // BAND_SCHEDULER_fixed_worker_idle_scheduler_H_