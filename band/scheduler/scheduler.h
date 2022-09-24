#ifndef BAND_SCHEDULER_SCHEDULER_H_
#define BAND_SCHEDULER_SCHEDULER_H_

#include <map>

#include "band/c/common.h"
#include "band/context.h"

/*
List of changes (compared to band-tflite)
1. Remove dependency to planner
2. Directly returns a set of schedule action
3. Gets scheduling context
4. Enforece
*/

namespace Band {
class Planner;
// Decision from a scheduler. The Jobs in the action must be passed to
// the appropriate workers.
using ScheduleAction =
    std::map<WorkerId, std::vector<std::pair<Job, SubgraphKey>>>;

class IScheduler {
public:
  IScheduler() = default;
  virtual ~IScheduler() = default;
  // A Schedule() function is expected to do the followings:
  // For the given requests, selected requests to schedule and
  // find the appropriate devices. The selected requests should be
  // enqueued in the `action` and removed from original queue.
  virtual ScheduleAction Schedule(const Context &context,
                                  JobQueue &requests) = 0;
  virtual bool NeedProfile() = 0;
  virtual bool NeedFallbackSubgraphs() = 0;
  virtual BandWorkerType GetWorkerType() = 0;

protected:
};
} // namespace Band

#endif