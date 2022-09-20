#ifndef TENSORFLOW_LITE_PLANNER_BASELINE_CONFIGURABLE_SCHEDULER_H_
#define TENSORFLOW_LITE_PLANNER_BASELINE_CONFIGURABLE_SCHEDULER_H_

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/planner/planner.h"

namespace tflite {
namespace impl {
class Interpreter;

// assigns requested model to devices in a Round-robin manner.
class BaselineConfigurableScheduler : public Scheduler {
 public:
  explicit BaselineConfigurableScheduler(Planner* planner) : Scheduler(planner) {
    need_profile_ = false;
    need_fallback_subgraphs_ = false;
    worker_type_ = kDeviceQueue;
  }
  void Schedule(JobQueue& requests) override;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_BASELINE_CONFIGURABLE_SCHEDULER_H_
