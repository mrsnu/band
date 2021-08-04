#include "tensorflow/lite/planner/least_slack_first_scheduler.h"

#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace impl {

void LeastSlackFirstScheduler::Schedule(JobQueue& requests) {
  std::set<TfLiteDeviceFlags> idle_devices = planner_->GetIdleDevices();
  if (idle_devices.empty()) {
    // no device is idle; wait for next iteration
    return;
  }
  DeviceWaitingTime device_waiting = GetDeviceWaitingTime();
  SortByDeadline(requests);
  for (auto it = requests.begin(); it != requests.end();) {
    
    // if selected device is empty,
    Subgraph* target_subgraph = GetInterpreter()->subgraph(target_subgraph_idx);
    if (target_subgraph->GetPrevSubgraph() == nullptr) {
      // only set these fields if this is the first subgraph of this model
      most_urgent_job.expected_latency = largest_shortest_latency;
    }
    EnqueueAction(most_urgent_job, target_subgraph);

    device_waiting[target_subgraph->GetKey().device_flag] +=
        largest_shortest_latency;
    // else, increase the interator
  }
}

void LeastSlackFirstScheduler::SortByDeadline(JobQueue& requests) {

}

}  // namespace impl
}  // namespace tflite
