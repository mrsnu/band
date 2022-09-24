#ifndef BAND_SCHEDULER_FIXED_DEVICE_SCHEDULER_H_
#define BAND_SCHEDULER_FIXED_DEVICE_SCHEDULER_H_

#include "band/scheduler/scheduler.h"

namespace Band {

// assigns requested model to devices according to model_id.
class FixedDeviceScheduler : public IScheduler {
public:
  ScheduleAction Schedule(const Context &context, JobQueue &requests) override;
  bool NeedProfile() override { return false; }
  bool NeedFallbackSubgraphs() override { return false; }
  BandWorkerType GetWorkerType() override { return kDeviceQueue; }
};

class FixedDeviceGlobalQueueScheduler : public IScheduler {
public:
  ScheduleAction Schedule(const Context &context, JobQueue &requests) override;
  // Required for checking SLO violation.
  // We could add an option to this planner for skipping the SLO check,
  // in which case this function can return false.
  bool NeedProfile() override { return true; }
  bool NeedFallbackSubgraphs() override { return false; }
  BandWorkerType GetWorkerType() override { return kGlobalQueue; }
};

} // namespace Band

#endif // BAND_SCHEDULER_FIXED_DEVICE_SCHEDULER_H_
