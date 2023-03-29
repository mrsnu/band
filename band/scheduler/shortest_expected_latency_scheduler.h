#ifndef BAND_SCHEDULER_SHORTEST_EXPECTED_LATENCY_SCHEDULER_H_
#define BAND_SCHEDULER_SHORTEST_EXPECTED_LATENCY_SCHEDULER_H_

#include "band/scheduler/scheduler.h"

namespace Band {

class ShortestExpectedLatencyScheduler : public IScheduler {
 public:
  explicit ShortestExpectedLatencyScheduler(Context& context, int window_size);

  void Schedule(JobQueue& requests) override;
  bool NeedProfile() override { return true; }
  bool NeedFallbackSubgraphs() override { return true; }
  WorkerType GetWorkerType() override { return WorkerType::DeviceQueue; }

 private:
  const int window_size_;
};

}  // namespace Band

#endif  // BAND_SCHEDULER_SHORTEST_EXPECTED_LATENCY_SCHEDULER_H_
