#ifndef BAND_SCHEDULER_ROUND_ROBIN_IDLE_SCHEDULER_H_
#define BAND_SCHEDULER_ROUND_ROBIN_IDLE_SCHEDULER_H_

#include "band/scheduler/scheduler.h"

namespace band {

// assigns requested model to devices in a Round-robin manner.
class RoundRobinIdleScheduler : public IScheduler {
 public:
  using IScheduler::IScheduler;
  bool Schedule(JobQueue& requests) override;
  bool NeedFallbackSubgraphs() override { return false; }
  WorkerType GetWorkerType() override { return WorkerType::kDeviceQueue; }

 private:
  int idle_us_ = 5000;
};

}  // namespace band

#endif  // BAND_SCHEDULER_ROUND_ROBIN_SCHEDULER_H_
