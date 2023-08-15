#ifndef BAND_SCHEDULER_HETEROGENEOUS_EARLIEST_FINISH_TIME_SCHEDULER_H_
#define BAND_SCHEDULER_HETEROGENEOUS_EARLIEST_FINISH_TIME_SCHEDULER_H_

#include "band/scheduler/scheduler.h"

namespace band {

class HEFTScheduler : public IScheduler {
 public:
  explicit HEFTScheduler(IEngine& engine, int window_size, bool reserve);

  bool Schedule(JobQueue& requests) override;
  bool NeedFallbackSubgraphs() override { return true; }
  WorkerType GetWorkerType() override { return WorkerType::kGlobalQueue; }

 private:
  // job_id --> subgraph_key
  std::map<int, SubgraphKey> reserved_;
  const int window_size_;
  const bool reserve_;
};

}  // namespace band

#endif  // BAND_SCHEDULER_HETEROGENEOUS_EARLIEST_FINISH_TIME_SCHEDULER_H_
