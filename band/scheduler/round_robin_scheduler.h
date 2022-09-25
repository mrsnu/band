#ifndef BAND_SCHEDULER_ROUND_ROBIN_SCHEDULER_H_
#define BAND_SCHEDULER_ROUND_ROBIN_SCHEDULER_H_

#include "band/scheduler/scheduler.h"

namespace Band {

// assigns requested model to devices in a Round-robin manner.
class RoundRobinScheduler : public IScheduler {
public:
  ScheduleAction Schedule(const Context &context, JobQueue &requests) override;
  bool NeedProfile() override { return false; }
  bool NeedFallbackSubgraphs() override { return false; }
  BandWorkerType GetWorkerType() override { return kBandDeviceQueue; }
};

} // namespace Band

#endif // BAND_SCHEDULER_ROUND_ROBIN_SCHEDULER_H_
