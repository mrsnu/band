#ifndef BAND_SCHEDULER_SHORTEST_EXPECTED_LATENCY_SCHEDULER_H_
#define BAND_SCHEDULER_SHORTEST_EXPECTED_LATENCY_SCHEDULER_H_

#include "band/scheduler/scheduler.h"

namespace band {

class ShortestExpectedLatencyScheduler : public IScheduler {
 public:
  explicit ShortestExpectedLatencyScheduler(IEngine& engine, int window_size);

  bool Schedule(JobQueue& requests) override;
  bool NeedProfile() override { return true; }
  bool NeedFallbackSubgraphs() override { return true; }
  WorkerType GetWorkerType() override { return WorkerType::kGlobalQueue; }
  std::set<EstimatorType> GetEstimatorTypes() override {
    return {EstimatorType::kLatency};
  }

 private:
  const int window_size_;
};

}  // namespace band

#endif  // BAND_SCHEDULER_SHORTEST_EXPECTED_LATENCY_SCHEDULER_H_
