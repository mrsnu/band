#include "tensorflow/lite/planner/fixed_device_scheduler.h"

namespace tflite {
namespace impl {

void FixedDeviceScheduler::Schedule(JobQueue& requests) {
  // TODO: fallback subgraphs for FixedDevicePlanner?
  while (!requests.empty()) {
    Job to_execute = requests.front();
    requests.pop_front();

    int model_id = to_execute.model_id;
    int device_idx;
    if (kTfLiteCPU <= to_execute.device_id &&
        to_execute.device_id < kTfLiteNumDevices) {
      device_idx = to_execute.device_id;
    } else {
      device_idx = planner_->GetModelDeviceMap()[model_id];
    }

    int subgraph_idx = GetInterpreter()->GetSubgraphIdx(
        model_id, static_cast<TfLiteDeviceFlags>(device_idx));
    Subgraph* subgraph = GetInterpreter()->subgraph(subgraph_idx);
    EnqueueAction(to_execute, subgraph);
  }
}

}  // namespace impl
}  // namespace tflite
