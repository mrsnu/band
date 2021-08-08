#ifndef TENSORFLOW_LITE_PLANNER_LEAST_SLACK_FIRST_SCHEDULER_H_
#define TENSORFLOW_LITE_PLANNER_LEAST_SLACK_FIRST_SCHEDULER_H_

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/planner/planner.h"

namespace tflite {
namespace impl {

class LeastSlackFirstScheduler : public Scheduler {
 public:
  explicit LeastSlackFirstScheduler(Planner* planner) : Scheduler(planner) {
    need_profile_ = true;
    need_fallback_subgraphs_ = true;
    worker_type_ = kGlobalQueue;
  }

 private:
  void Schedule(JobQueue& requests) override;
  int64_t GetSlackTime(const Job& job);
  void SortBySlackTime(JobQueue& requests);
  void UpdateExpectedLatency(JobQueue& requests);
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_LEAST_SLACK_FIRST_SCHEDULER_H_
