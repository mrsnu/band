#ifndef TENSORFLOW_LITE_PLANNER_MOBILE_CLOUD_HEFT_SCHEDULER_H_
#define TENSORFLOW_LITE_PLANNER_MOBILE_CLOUD_HEFT_SCHEDULER_H_

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/planner/planner.h"

namespace tflite {
namespace impl {
class Interpreter;

class MobileCloudHeftScheduler : public Scheduler {
 public:
  explicit MobileCloudHeftScheduler(Planner* planner) : Scheduler(planner) {
    need_profile_ = true;
    worker_type_ = kGlobalQueue;
  }
  void Schedule(JobQueue& requests) override;

 private:
  std::map<int, int> reserved_;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_MOBILE_CLOUD_HEFT_SCHEDULER_H_
