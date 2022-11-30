#ifndef TENSORFLOW_LITE_PLANNER_MOBILE_CLOUD_HEFT_SCHEDULER_H_
#define TENSORFLOW_LITE_PLANNER_MOBILE_CLOUD_HEFT_SCHEDULER_H_

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/planner/planner.h"

namespace tflite {
namespace impl {

class MobileCloudHeftScheduler : public Scheduler {
 public:
  explicit MobileCloudHeftScheduler(Planner* planner, ModelManager* model_manager) : Scheduler(planner) {
    need_profile_ = false; // It's only for warm up
    worker_type_ = kDeviceQueue;
    model_manager_ = model_manager;
  }
  void Schedule(JobQueue& requests) override;

 private:
  ModelManager * model_manager_;

  std::map<int, int> reserved_;
  std::pair<int, int64_t> GetShortestSubgraph(int model_id, std::map<int, int64_t>& worker_waiting);
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_MOBILE_CLOUD_HEFT_SCHEDULER_H_