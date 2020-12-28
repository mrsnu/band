#ifndef TENSORFLOW_LITE_FIXED_DEVICE_PLANNER_H_
#define TENSORFLOW_LITE_FIXED_DEVICE_PLANNER_H_

#include "tensorflow/lite/planner.h"
#include "tensorflow/lite/interpreter.h"

namespace tflite {

namespace impl {

// assigns requested model to devices according to model_id.
class FixedDevicePlanner : public Planner {
  public:
    explicit FixedDevicePlanner(Interpreter* interpreter)
      : Planner(interpreter) {
      planner_thread_ = std::thread([this]{this->Plan();});
    }
    void Plan() override;
    bool NeedProfile() override;
  private:
    // Map structure to find assigned device of model idx (model_id, device flag)
    std::map<int, int> model_device_map;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_FIXED_DEVICE_PLANNER_H_
