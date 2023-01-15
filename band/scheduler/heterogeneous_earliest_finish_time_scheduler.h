#ifndef BAND_SCHEDULER_HETEROGENEOUS_EARLIEST_FINISH_TIME_SCHEDULER_H_
#define BAND_SCHEDULER_HETEROGENEOUS_EARLIEST_FINISH_TIME_SCHEDULER_H_

#include "band/scheduler/scheduler.h"

namespace Band {

class HeterogeneousEarliestFinishTimeScheduler : public IScheduler {
 public:
  ScheduleAction Schedule(const Context& context) override;
  bool NeedProfile() override { return true; }
  bool NeedFallbackSubgraphs() override { return true; }
  BandWorkerType GetWorkerType() override { return kBandGlobalQueue; }
};

}  // namespace Band

#endif  // BAND_SCHEDULER_HETEROGENEOUS_EARLIEST_FINISH_TIME_SCHEDULER_H_
