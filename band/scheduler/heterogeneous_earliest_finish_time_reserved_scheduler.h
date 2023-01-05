#ifndef BAND_SCHEDULER_HETEROGENEOUS_EARLIEST_FINISH_TIME_RESERVED_SCHEDULER_H_
#define BAND_SCHEDULER_HETEROGENEOUS_EARLIEST_FINISH_TIME_RESERVED_SCHEDULER_H_

#include "band/scheduler/scheduler.h"

namespace Band {

class HEFTReservedScheduler : public IScheduler {
 public:
  explicit HEFTReservedScheduler(int window_size);

  ScheduleAction Schedule(const Context& context, JobQueue& requests) override;
  bool NeedProfile() override { return true; }
  bool NeedFallbackSubgraphs() override { return true; }
  BandWorkerType GetWorkerType() override { return kBandGlobalQueue; }

 private:
  // job_id --> subgraph_idx
  std::map<int, SubgraphKey> reserved_;
  const int window_size_;
};

}  // namespace Band

#endif  // BAND_SCHEDULER_HETEROGENEOUS_EARLIEST_FINISH_TIME_RESERVED_SCHEDULER_H_
