#ifndef TENSORFLOW_LITE_PLANNER_RANDOM_ASSIGN_SCHEDULER_H_
#define TENSORFLOW_LITE_PLANNER_RANDOM_ASSIGN_SCHEDULER_H_

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/planner/planner.h"

namespace tflite {
namespace impl {
class Interpreter;

// assigns requested model to devices in a random manner.
class RandomAssignScheduler : public Scheduler {
 public:
  explicit RandomAssignScheduler(Planner* planner, ModelManager* model_manager) : Scheduler(planner) {
    need_profile_ = false;
    worker_type_ = kDeviceQueue;
    model_manager_ = model_manager;
  }
  void Schedule(JobQueue& requests) override;
 private:
  ModelManager * model_manager_;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_RANDOM_ASSIGN_SCHEDULER_H_
