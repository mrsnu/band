#ifndef BAND_SCHEDULER_DVFS_SCHEDULER_H_
#define BAND_SCHEDULER_DVFS_SCHEDULER_H_

#include "band/scheduler/scheduler.h"

namespace band {

class DVFSScheduler : public IScheduler {
 public:
  using IScheduler::IScheduler;
  bool Schedule(JobQueue& requests) override;
  bool NeedFallbackSubgraphs() override { return false; }
  WorkerType GetWorkerType() override { return WorkerType::kDeviceQueue; }
};

}  // namespace band

#endif  // BAND_SCHEDULER_DVFS_SCHEDULER_H_