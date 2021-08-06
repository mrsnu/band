#include "tensorflow/lite/planner/fixed_device_scheduler.h"
#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace impl {

void FixedDeviceGlobalQueueScheduler::Schedule(JobQueue& requests) {
  // TODO: fallback subgraphs for FixedDevicePlanner?
  std::set<TfLiteDeviceFlags> idle_devices = planner_->GetIdleDevices();
  if (idle_devices.empty()) {
    // no device is idle; wait for next iteration
    return;
  }

  for (auto it = requests.begin(); it != requests.end();) {
    Job& to_execute = *it;
    int model_id = to_execute.model_id;

    int device_idx;
    if (kTfLiteCPU <= to_execute.device_id &&
        to_execute.device_id < kTfLiteNumDevices) {
      device_idx = to_execute.device_id;
    } else {
      device_idx = planner_->GetModelDeviceMap()[model_id];
    }
    TfLiteDeviceFlags device_flag = static_cast<TfLiteDeviceFlags>(device_idx);

    auto idle_devices_it = idle_devices.find(device_flag);
    if (idle_devices_it == idle_devices.end()) {
      // that device is not idle, so leave this job alone for now
      ++it;
      continue;
    }

    int subgraph_idx = GetInterpreter()->GetSubgraphIdx(model_id, device_flag);
    Subgraph* subgraph = GetInterpreter()->subgraph(subgraph_idx);
    to_execute.expected_latency =
        GetDeviceWaitingTime()[device_flag] +
        GetInterpreter()->GetExpectedLatency(subgraph->GetKey());
    EnqueueAction(to_execute, subgraph);

    // delete this job from our request queue and
    // delete this device from our idle_devices set
    it = requests.erase(it);
    idle_devices.erase(idle_devices_it);

    if (idle_devices.empty()) {
      // no device is idle; wait for next iteration
      break;
    }
  }
}

}  // namespace impl
}  // namespace tflite
