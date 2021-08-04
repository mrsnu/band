#include "tensorflow/lite/planner/round_robin_scheduler.h"

namespace tflite {
namespace impl {

ScheduleAction RoundRobinScheduler::Schedule(JobQueue& requests) {
  ScheduleAction action;
  std::set<TfLiteDeviceFlags> idle_devices = planner_->GetIdleDevices();
  for (auto device : idle_devices) {
    if (!requests.empty()) {
      auto available_job = std::find_if(
          requests.begin(), requests.end(), [this, device](const Job& job) {
            return GetInterpreter()->GetSubgraphIdx(job.model_id, device) != -1;
          });
      if (available_job != requests.end()) {
        Job to_execute = *available_job;
        /*
        int subgraph_idx =
            GetInterpreter()->GetSubgraphIdx(to_execute.model_id, device);
        SubgraphKey& key = GetInterpreter()->subgraph(subgraph_idx)->GetKey();
        planner_->UpdateJobEnqueueStatus(to_execute, key);
        */

        action[device].push_back(to_execute);
        requests.erase(available_job);
      }
    }
  }
  return action;
}

}  // namespace impl
}  // namespace tflite
