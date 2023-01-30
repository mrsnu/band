#ifndef BAND_SCHEDULER_LEAST_SLACK_FIRST_SCHEDULER_H_
#define BAND_SCHEDULER_LEAST_SLACK_FIRST_SCHEDULER_H_

#include "band/scheduler/scheduler.h"

namespace Band {

class LeastSlackFirstScheduler : public IScheduler {
 public:
  explicit LeastSlackFirstScheduler(Context* context, int window_size);

  void Schedule(JobQueue& requests) override;
  bool NeedProfile() override { return true; }
  bool NeedFallbackSubgraphs() override { return true; }
  BandWorkerType GetWorkerType() override { return kBandGlobalQueue; }

 private:
  int64_t GetSlackTime(int64_t current_time, const Job& job);
  void SortBySlackTime(JobQueue& requests_, int window_size,
                       int64_t current_time);
  void UpdateExpectedLatency(JobQueue& requests_, int window_size);
  const int window_size_;
};

}  // namespace Band

#endif  // BAND_SCHEDULER_LEAST_SLACK_FIRST_SCHEDULER_H_
