#ifndef TENSORFLOW_LITE_PLANNER_FIXED_DEVICE_SCHEDULER_H_
#define TENSORFLOW_LITE_PLANNER_FIXED_DEVICE_SCHEDULER_H_

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/planner/planner.h"
#include "tensorflow/lite/planner/util.h"

namespace tflite {

namespace impl {

// assigns requested model to devices according to model_id.
class FixedDeviceScheduler : public Scheduler {
 public:
  explicit FixedDeviceScheduler(Planner* planner) : Scheduler(planner) {
    need_profile_ = false;
    worker_type_ = kDeviceQueue;
  }
  void Schedule(JobQueue& requests) override;
};

class FixedDeviceGlobalQueueScheduler : public Scheduler {
 public:
  explicit FixedDeviceGlobalQueueScheduler(Planner* planner)
      : Scheduler(planner) {
    // Required for checking SLO violation.
    // We could add an option to this planner for skipping the SLO check,
    // in which case this function can return false.
    need_profile_ = true;
    worker_type_ = kGlobalQueue;
  }
  void Schedule(JobQueue& requests) override;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_FIXED_DEVICE_SCHEDULER_H_
