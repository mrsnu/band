#include "tensorflow/lite/planner/fixed_device_scheduler.h"

namespace tflite {
namespace impl {

ScheduleAction FixedDeviceScheduler::Schedule(JobQueue& requests) {
  ScheduleAction action;
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

    /*
    TfLiteDeviceFlags device_flag = static_cast<TfLiteDeviceFlags>(device_idx);
    // TODO: fallback subgraphs for FixedDevicePlanner?
    int subgraph_idx = GetInterpreter()->GetSubgraphIdx(model_id, device_flag);
    SubgraphKey& key = GetInterpreter()->subgraph(subgraph_idx)->GetKey();
    planner_->UpdateJobEnqueueStatus(to_execute, key);
    */

    action[device_flag].push_back(to_execute);
  }
  return action;
}

}  // namespace impl
}  // namespace tflite
