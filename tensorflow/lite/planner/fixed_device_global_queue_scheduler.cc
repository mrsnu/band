#include "tensorflow/lite/planner/fixed_device_scheduler.h"
#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace impl {

ScheduleAction FixedDeviceGlobalQueueScheduler::Schedule(JobQueue& requests) {
  ScheduleAction action;
  std::set<TfLiteDeviceFlags> idle_devices = planner_->GetIdleDevices();
  if (idle_devices.empty()) {
    // no device is idle; wait for next iteration
    return action;
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

    /*
    // TODO: fallback subgraphs for FixedDevicePlanner?
    int subgraph_idx = GetInterpreter()->GetSubgraphIdx(model_id, device_flag);
    // Record expected latency to check if the SLO has been violated.
    SubgraphKey& key = GetInterpreter()->subgraph(subgraph_idx)->GetKey();
    int64_t profiled = GetInterpreter()->GetExpectedLatency(key);
    int64_t expected_latency = GetDeviceWaitingTime()[device_flag] + profiled;
    to_execute.expected_latency = expected_latency;
    planner_->UpdateJobEnqueueStatus(to_execute, key);
    */

    action[device_flag].push_back(to_execute);

    // delete this job from our request queue and
    // delete this device from our idle_devices set
    it = requests.erase(it);
    idle_devices.erase(idle_devices_it);

    if (idle_devices.empty()) {
      // no device is idle; wait for next iteration
      break;
    }
  }
  return action;
}

}  // namespace impl
}  // namespace tflite
