#ifndef BAND_SCHEDULER_FIXED_WORKER_SCHEDULER_H_
#define BAND_SCHEDULER_FIXED_WORKER_SCHEDULER_H_

#include "band/scheduler/scheduler.h"

namespace band {

// Assigns requested model to devices according to a direct request from engine
// or model_id.
class FixedWorkerScheduler : public IScheduler {
 public:
  using IScheduler::IScheduler;
  bool Schedule(JobQueue& requests) override;
  bool NeedFallbackSubgraphs() override { return false; }
  WorkerType GetWorkerType() override { return WorkerType::kDeviceQueue; }
};

class FixedWorkerGlobalQueueScheduler : public IScheduler {
 public:
  using IScheduler::IScheduler;
  bool Schedule(JobQueue& requests) override;
  bool NeedFallbackSubgraphs() override { return false; }
  WorkerType GetWorkerType() override { return WorkerType::kGlobalQueue; }
};

}  // namespace band

#endif  // BAND_SCHEDULER_fixed_worker_scheduler_H_
