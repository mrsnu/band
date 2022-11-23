#ifndef TENSORFLOW_LITE_PLANNER_THERMAL_AWARE_SCHEDULER_H_
#define TENSORFLOW_LITE_PLANNER_THERMAL_AWARE_SCHEDULER_H_

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/planner/planner.h"

namespace tflite {
namespace impl {

class ThermalAwareScheduler : public Scheduler {
 public:
  explicit ThermalAwareScheduler(Planner* planner, ModelManager* model_manager) : Scheduler(planner) {
    need_profile_ = true;
    worker_type_ = kDeviceQueue;
    model_manager_ = model_manager;
  }
  void Schedule(JobQueue& requests) override;

 private:
  ModelManager * model_manager_;

  std::pair<int, double> GetMaxPptSubgraphIdx(Job& job, std::map<int, int64_t>& worker_waiting);
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_THERMAL_AWARE_SCHEDULER_H_
