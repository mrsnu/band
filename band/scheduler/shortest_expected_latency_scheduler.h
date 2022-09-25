#ifndef BAND_SCHEDULER_SHORTEST_EXPECTED_LATENCY_SCHEDULER_H_
#define BAND_SCHEDULER_SHORTEST_EXPECTED_LATENCY_SCHEDULER_H_

#include "band/scheduler/scheduler.h"

namespace Band {

class ShortestExpectedLatencyScheduler : public IScheduler {
public:
  ScheduleAction Schedule(const Context &context) override;
  bool NeedProfile() override { return true; }
  bool NeedFallbackSubgraphs() override { return true; }
  BandWorkerType GetWorkerType() override { return kBandDeviceQueue; }
};

} // namespace Band

#endif // BAND_SCHEDULER_SHORTEST_EXPECTED_LATENCY_SCHEDULER_H_
