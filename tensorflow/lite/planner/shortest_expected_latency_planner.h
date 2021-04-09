#ifndef TENSORFLOW_LITE_PLANNER_SHORTEST_EXPECTED_LATENCY_PLANNER_H_
#define TENSORFLOW_LITE_PLANNER_SHORTEST_EXPECTED_LATENCY_PLANNER_H_

#include "tensorflow/lite/planner/planner.h"
#include "tensorflow/lite/interpreter.h"

namespace tflite {
namespace impl {

class ShortestExpectedLatencyPlanner : public Planner {
 public:
  explicit ShortestExpectedLatencyPlanner(Interpreter* interpreter,
                                          int schedule_window_size)
      : Planner(interpreter) {
    planner_thread_ = std::thread([this]{this->Plan();});
    schedule_window_size_ = schedule_window_size;
  }
  void Plan() override;
  bool NeedProfile() override;

  int GetWindowSize() {
    return schedule_window_size_;
  }

  private:
    int schedule_window_size_ = INT_MAX;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_SHORTEST_EXPECTED_LATENCY_PLANNER_H_
