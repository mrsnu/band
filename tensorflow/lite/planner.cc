#include "tensorflow/lite/planner.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"

namespace tflite {
namespace impl {

TfLiteStatus Planner::Plan(Interpreter* interpreter) {
  if (!change_plan_) return kTfLiteOk;
  change_plan_ = false;
  TfLiteStatus status;

  for (int i = 0; i < interpreter->subgraphs_size(); ++i) {
    Subgraph& subgraph = *(interpreter->subgraph(i));
    status = subgraph.UndoAllDelegates();
    if (status != kTfLiteOk)
      return status;

    if (i % kTfLiteNumDevices == 1) {
      subgraph.SetModelPlan(kTfLiteGPU);
    } else if (i % kTfLiteNumDevices == 2) {
      subgraph.SetModelPlan(kTfLiteDSP);
    }

    status = subgraph.ModifyGraphWithDelegate(
        interpreter->device_delegates(subgraph.GetModelPlan()));
    if (status != kTfLiteOk)
      return status;
  }

  return kTfLiteOk;
}

}  // namespace impl
}  // namespace tflite
