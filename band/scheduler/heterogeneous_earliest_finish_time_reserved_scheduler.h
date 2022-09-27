#ifndef BAND_SCHEDULER_HETEROGENEOUS_EARLIEST_FINISH_TIME_RESERVED_SCHEDULER_H_
#define BAND_SCHEDULER_HETEROGENEOUS_EARLIEST_FINISH_TIME_RESERVED_SCHEDULER_H_

#include "band/scheduler/scheduler.h"

namespace Band {

class HeterogeneousEarliestFinishTimeReservedScheduler : public IScheduler {
 public:
  ScheduleAction Schedule(const Context& context) override;
  bool NeedProfile() override { return true; }
  bool NeedFallbackSubgraphs() override { return true; }
  BandWorkerType GetWorkerType() override { return kBandGlobalQueue; }

 private:
  // job_id --> subgraph_idx
  std::map<int, int> reserved_;
};

}  // namespace Band

#endif  // BAND_SCHEDULER_HETEROGENEOUS_EARLIEST_FINISH_TIME_RESERVED_SCHEDULER_H_
