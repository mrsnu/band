#ifndef BAND_SCHEDULER_LEAST_SLACK_FIRST_SCHEDULER_H_
#define BAND_SCHEDULER_LEAST_SLACK_FIRST_SCHEDULER_H_

#include "band/scheduler/scheduler.h"

namespace band {

class LeastSlackFirstScheduler : public IScheduler {
 public:
  explicit LeastSlackFirstScheduler(IEngine& engine, int window_size);

  bool Schedule(JobQueue& requests) override;
  bool NeedProfile() override { return true; }
  bool NeedFallbackSubgraphs() override { return true; }
  WorkerType GetWorkerType() override { return WorkerType::GlobalQueue; }

 private:
  int64_t GetSlackTime(int64_t current_time, const Job& job);
  void SortBySlackTime(JobQueue& requests, int window_size,
                       int64_t current_time);
  void UpdateExpectedLatency(JobQueue& requests, int window_size);
  const int window_size_;
};

}  // namespace band

#endif  // BAND_SCHEDULER_LEAST_SLACK_FIRST_SCHEDULER_H_
