#ifndef TENSORFLOW_LITE_PLANNER_HETEROGENEOUS_EARLIEST_FINISH_TIME_SCHEDULER_H_
#define TENSORFLOW_LITE_PLANNER_HETEROGENEOUS_EARLIEST_FINISH_TIME_SCHEDULER_H_

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/planner/planner.h"

namespace tflite {
namespace impl {

class HeterogeneousEarliestFinishTimeScheduler : public Scheduler {
 public:
  explicit HeterogeneousEarliestFinishTimeScheduler(Planner* planner)
      : Scheduler(planner) {
    need_profile_ = true;
    need_fallback_subgraphs_ = true;
    worker_type_ = kGlobalQueue;
  }
  void Schedule(JobQueue& requests) override;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_HETEROGENEOUS_EARLIEST_FINISH_TIME_SCHEDULER_H_
