#ifndef TENSORFLOW_LITE_PLANNER_ROUND_ROBIN_PLANNER_H_
#define TENSORFLOW_LITE_PLANNER_ROUND_ROBIN_PLANNER_H_

#include "tensorflow/lite/planner/planner.h"
#include "tensorflow/lite/interpreter.h"

namespace tflite {

namespace impl {

class Interpreter;

// assigns requested model to devices in a Round-robin manner.
class RoundRobinPlanner : public Planner {
 public:
  explicit RoundRobinPlanner(Interpreter* interpreter, std::string log_path)
    : Planner(interpreter, log_path) {
      planner_thread_ = std::thread([this]{this->Plan();});
  }
  void Plan() override;
  bool NeedProfile() override;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_ROUND_ROBIN_PLANNER_H_
