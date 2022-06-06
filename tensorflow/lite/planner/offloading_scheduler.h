#ifndef TENSORFLOW_LITE_PLANNER_OFFLOADING_SCHEDULER_H_
#define TENSORFLOW_LITE_PLANNER_OFFLOADING_SCHEDULER_H_

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/planner/planner.h"

namespace tflite {
namespace impl {

class OffloadingScheduler : public Scheduler {
 public:
  explicit OffloadingScheduler(Planner* planner)
      : Scheduler(planner) {
    need_profile_ = true;
    need_fallback_subgraphs_ = true;
    worker_type_ = kDeviceQueue;
  }
  void Schedule(JobQueue& requests) override;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_OFFLOADING_SCHEDULER_H_
