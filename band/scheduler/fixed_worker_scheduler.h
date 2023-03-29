#ifndef BAND_SCHEDULER_FIXED_WORKER_SCHEDULER_H_
#define BAND_SCHEDULER_FIXED_WORKER_SCHEDULER_H_

#include "band/scheduler/scheduler.h"

namespace Band {

// Assigns requested model to devices according to a direct request from engine
// or model_id.
class FixedWorkerScheduler : public IScheduler {
 public:
  using IScheduler::IScheduler;
  void Schedule(JobQueue& requests) override;
  bool NeedProfile() override { return false; }
  bool NeedFallbackSubgraphs() override { return false; }
  WorkerType GetWorkerType() override { return WorkerType::DeviceQueue; }
};

class FixedWorkerGlobalQueueScheduler : public IScheduler {
 public:
  using IScheduler::IScheduler;
  void Schedule(JobQueue& requests) override;
  // Required for checking SLO violation.
  // We could add an option to this planner for skipping the SLO check,
  // in which case this function can return false.
  bool NeedProfile() override { return true; }
  bool NeedFallbackSubgraphs() override { return false; }
  WorkerType GetWorkerType() override { return WorkerType::GlobalQueue; }
};

}  // namespace Band

#endif  // BAND_SCHEDULER_fixed_worker_scheduler_H_
