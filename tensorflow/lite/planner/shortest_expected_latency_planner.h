#ifndef TENSORFLOW_LITE_PLANNER_SHORTEST_EXPECTED_LATENCY_PLANNER_H_
#define TENSORFLOW_LITE_PLANNER_SHORTEST_EXPECTED_LATENCY_PLANNER_H_

#include "tensorflow/lite/planner/planner.h"
#include "tensorflow/lite/interpreter.h"

namespace tflite {
namespace impl {

class ShortestExpectedLatencyScheduler : public Scheduler {
 public:
  explicit ShortestExpectedLatencyScheduler(Planner* planner)
      : Scheduler(planner) {
    need_profile_ = true;
    worker_type_ = DeviceQueue;
  }
  ScheduleAction Schedule(JobQueue& requests) override;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_SHORTEST_EXPECTED_LATENCY_PLANNER_H_
