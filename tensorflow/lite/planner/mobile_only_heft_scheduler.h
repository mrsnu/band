#ifndef TENSORFLOW_LITE_PLANNER_MOBILE_ONLY_HEFT_SCHEDULER_H_
#define TENSORFLOW_LITE_PLANNER_MOBILE_ONLY_HEFT_SCHEDULER_H_

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/planner/planner.h"

namespace tflite {
namespace impl {

class MobileOnlyHeftScheduler : public Scheduler {
 public:
  explicit MobileOnlyHeftScheduler(Planner* planner)
      : Scheduler(planner) {
    need_profile_ = true;
    worker_type_ = kGlobalQueue;
  }
  void Schedule(JobQueue& requests) override;

 private:
  // job_id --> subgraph_idx
  std::map<int, int> reserved_;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_MOBILE_ONLY_HEFT_SCHEDULER_H_