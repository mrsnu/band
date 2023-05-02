#ifndef BAND_SCHEDULER_SCHEDULER_H_
#define BAND_SCHEDULER_SCHEDULER_H_

#include <map>

#include "band/context.h"

namespace band {
class Planner;

class IScheduler {
 public:
  explicit IScheduler(Context& context) : context_(context) {}
  virtual ~IScheduler() = default;
  // A Schedule() function is expected to do the followings:
  // For the given requests, selected requests to schedule and
  // find the appropriate devices. The selected requests should be
  // enqueued to the worker and removed from original queue.
  virtual absl::Status Schedule(JobQueue& requests) = 0;
  virtual bool NeedProfile() = 0;
  virtual bool NeedFallbackSubgraphs() = 0;
  virtual WorkerType GetWorkerType() = 0;
  virtual const ErrorReporter* GetErrorReporter() {
    return context_.GetErrorReporter();
  }

 protected:
  Context& context_;
};
}  // namespace band

#endif