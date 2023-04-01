#ifndef BAND_SCHEDULER_ROUND_ROBIN_SCHEDULER_H_
#define BAND_SCHEDULER_ROUND_ROBIN_SCHEDULER_H_

#include "band/scheduler/scheduler.h"

namespace band {

// assigns requested model to devices in a Round-robin manner.
class RoundRobinScheduler : public IScheduler {
 public:
  using IScheduler::IScheduler;
  void Schedule(JobQueue& requests) override;
  bool NeedProfile() override { return false; }
  bool NeedFallbackSubgraphs() override { return false; }
  WorkerType GetWorkerType() override { return WorkerType::DeviceQueue; }
};

}  // namespace band

#endif  // BAND_SCHEDULER_ROUND_ROBIN_SCHEDULER_H_
