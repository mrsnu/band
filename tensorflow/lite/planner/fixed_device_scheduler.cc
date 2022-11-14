#include "tensorflow/lite/planner/fixed_device_scheduler.h"

namespace tflite {
namespace impl {

void FixedDeviceScheduler::Schedule(JobQueue& requests) {
  // TODO: fallback subgraphs for FixedDevicePlanner?
  while (!requests.empty()) {
    Job to_execute = requests.front();
    requests.pop_front();

    int model_id = to_execute.model_id;
    int worker_id = to_execute.model_id;
    // if (0 <= to_execute.device_id && to_execute.device_id < kTfLiteNumDevices) {
    //   TfLiteDeviceFlags device_flag = static_cast<TfLiteDeviceFlags>(to_execute.device_id);
    //   worker_id = GetInterpreter()->GetRepresentativeWorkerId(device_flag);
    // } else {
    //   worker_id = planner_->GetModelWorkerMap()[model_id];
    // }

    int subgraph_idx = GetInterpreter()->GetSubgraphIdx(
        model_id, worker_id);
    Subgraph* subgraph = GetInterpreter()->subgraph(subgraph_idx);
    EnqueueAction(to_execute, subgraph);
  }
}

}  // namespace impl
}  // namespace tflite
