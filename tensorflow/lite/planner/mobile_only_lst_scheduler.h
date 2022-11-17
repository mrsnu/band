#ifndef TENSORFLOW_LITE_PLANNER_MOBILE_ONLY_LST_H_
#define TENSORFLOW_LITE_PLANNER_MOBILE_ONLY_LST_H_

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/planner/planner.h"

namespace tflite {
namespace impl {

class MobileOnlyLstScheduler : public Scheduler {
 public:
  explicit MobileOnlyLstScheduler(Planner* planner)
      : Scheduler(planner) {
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

#endif  // TENSORFLOW_LITE_PLANNER_MOBILE_ONLY_LST_H_