#ifndef BAND_SCHEDULER_FRAME_THERMAL_SCHEDULER_H_
#define BAND_SCHEDULER_FRAME_THERMAL_SCHEDULER_H_

#include "band/scheduler/scheduler.h"

namespace band {

class FrameThremalScheduler : public IScheduler {
 public:
  explicit FrameThremalScheduler(IEngine& engine) : IScheduler(engine) {};

  bool Schedule(JobQueue& requests) override;
  bool NeedFallbackSubgraphs() override { return true; }
  WorkerType GetWorkerType() override { return WorkerType::kGlobalQueue; }

 private:
};

}  // namespace band

#endif  // BAND_SCHEDULER_FRAME_THERMAL_SCHEDULER_H_