#ifndef TENSORFLOW_LITE_PLANNER_CLOUD_ONLY_SCHEDULER_H
#define TENSORFLOW_LITE_PLANNER_CLOUD_ONLY_SCHEDULER_H

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/planner/planner.h"
#include "tensorflow/lite/planner/util.h"

namespace tflite {

namespace impl {

// assigns requested model to devices according to model_id.
class CloudOnlyScheduler : public Scheduler {
 public:
  explicit CloudOnlyScheduler(Planner* planner) : Scheduler(planner) {
    need_profile_ = false;
    worker_type_ = kDeviceQueue;
  }
  void Schedule(JobQueue& requests) override;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_CLOUD_ONLY_SCHEDULER_H
