#ifndef BAND_SCHEDULER_GREEDY_THERMAL_SCHEDULER_H_
#define BAND_SCHEDULER_GREEDY_THERMAL_SCHEDULER_H_

#include "band/scheduler/scheduler.h"

namespace band {

class GreedyThermalScheduler : public IScheduler {
 public:
  explicit GreedyThermalScheduler(IEngine& engine) : IScheduler(engine){};

  bool Schedule(JobQueue& requests) override;
  bool NeedFallbackSubgraphs() override { return true; }
  WorkerType GetWorkerType() override { return WorkerType::kGlobalQueue; }

 private:
  std::pair<int, double> GetMinCostSubgraph(Job& job,
                                            WorkerWaitingTime& waiting_time);
};

}  // namespace band

#endif  // BAND_SCHEDULER_GREEDY_THERMAL_SCHEDULER_H_