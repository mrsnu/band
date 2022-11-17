#ifndef TENSORFLOW_LITE_PLANNER_MOBILE_CLOUD_LST_SCHEDULER_H_
#define TENSORFLOW_LITE_PLANNER_MOBILE_CLOUD_LST_SCHEDULER_H_

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/planner/planner.h"

namespace tflite {
namespace impl {
class Interpreter;

class MobileCloudLstScheduler : public Scheduler {
 public:
  explicit MobileCloudLstScheduler(Planner* planner) : Scheduler(planner) {
    need_profile_ = true;
    worker_type_ = kGlobalQueue;
  }
  void Schedule(JobQueue& requests) override;

 private:
  int64_t GetSlackTime(int64_t current_time, const Job& job);
  void SortBySlackTime(JobQueue& requests, int window_size,
                       int64_t current_time);
  void UpdateExpectedLatency(JobQueue& requests, int window_size);
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_MOBILE_CLOUD_LST_SCHEDULER_H_
