#ifndef BAND_SCHEDULER_LEAST_SLACK_FIRST_SCHEDULER_H_
#define BAND_SCHEDULER_LEAST_SLACK_FIRST_SCHEDULER_H_

#include "band/scheduler/scheduler.h"

namespace Band {

class LeastSlackFirstScheduler : public IScheduler {
public:
  explicit LeastSlackFirstScheduler(int window_size);

  ScheduleAction Schedule(const Context &context) override;
  bool NeedProfile() override { return true; }
  bool NeedFallbackSubgraphs() override { return false; }
  BandWorkerType GetWorkerType() override { return kBandGlobalQueue; }

private:
  int64_t GetSlackTime(int64_t current_time, const Job &job);
  void SortBySlackTime(const Context &context, JobQueue &requests_,
                       int window_size, int64_t current_time);
  void UpdateExpectedLatency(const Context &context, JobQueue &requests_,
                             int window_size);
  const int window_size_;
};

} // namespace Band

#endif // BAND_SCHEDULER_LEAST_SLACK_FIRST_SCHEDULER_H_
