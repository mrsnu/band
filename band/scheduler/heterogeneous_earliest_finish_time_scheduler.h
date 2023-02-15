#ifndef BAND_SCHEDULER_HETEROGENEOUS_EARLIEST_FINISH_TIME_SCHEDULER_H_
#define BAND_SCHEDULER_HETEROGENEOUS_EARLIEST_FINISH_TIME_SCHEDULER_H_

#include "band/scheduler/scheduler.h"

namespace band {

class HEFTScheduler : public IScheduler {
 public:
  explicit HEFTScheduler(Context& context, int window_size, bool reserve);

  void Schedule(JobQueue& requests) override;
  bool NeedProfile() override { return true; }
  bool NeedFallbackSubgraphs() override { return true; }
  BandWorkerType GetWorkerType() override { return kBandGlobalQueue; }

 private:
  // job_id --> subgraph_key
  std::map<int, SubgraphKey> reserved_;
  const int window_size_;
  const bool reserve_;
};

}  // namespace band

#endif  // BAND_SCHEDULER_HETEROGENEOUS_EARLIEST_FINISH_TIME_SCHEDULER_H_
