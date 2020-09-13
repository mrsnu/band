#ifndef TENSORFLOW_LITE_PLANNER_H_
#define TENSORFLOW_LITE_PLANNER_H_

#include <memory>
#if defined(__ANDROID__)
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#endif

namespace tflite {

namespace impl {

class Interpreter;
class Subgraph;

typedef enum {
  kTfLiteCPU = 0,
  kTfLiteGPU = 1,
  kTfLiteDSP = 2,
  kTfLiteNumDevices = 3,
} TfLiteDevice;

// Contains how a Subgraph should be executed.
// Currently, the unit of device placement is a `Subgraph`.
// Each Subgraph contains one `ModelPlan` as a member.
struct ModelPlan{
 public:
  ModelPlan():device_(kTfLiteCPU) {}
  ModelPlan(ModelPlan&&) = default;
  ModelPlan(const ModelPlan&) = delete;
  TfLiteDevice device_;
};

// assigns requested model to devices according to `ModelPlan` of a `Subgraph`.
// The interpreter manages a `Planner`.
class Planner{
 public:
  Planner() {}
  ~Planner() {}
  TfLiteStatus Plan(Interpreter* interpreter);
  bool change_plan_ = true;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_H_
