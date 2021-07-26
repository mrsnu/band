#ifndef TENSORFLOW_LITE_PLANNER_ROUND_ROBIN_SCHEDULER_H_
#define TENSORFLOW_LITE_PLANNER_ROUND_ROBIN_SCHEDULER_H_

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/planner/planner.h"

namespace tflite {

namespace impl {

class Interpreter;

// assigns requested model to devices in a Round-robin manner.
class RoundRobinScheduler : public Scheduler {
 public:
  explicit RoundRobinScheduler(Planner* planner) : Scheduler(planner) {
    need_profile_ = false;
    worker_type_ = kDeviceQueue;
  }
  ScheduleAction Schedule(JobQueue& requests) override;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_ROUND_ROBIN_SCHEDULER_H_
