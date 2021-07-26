#include "tensorflow/lite/planner/round_robin_scheduler.h"

namespace tflite {
namespace impl {

ScheduleAction RoundRobinScheduler::Schedule(JobQueue& requests) {
  ScheduleAction action;
  std::set<TfLiteDeviceFlags> idle_devices = planner_->GetIdleDevices();
  for(auto device : idle_devices) {
    while (!requests.empty()) {
      auto available_job =
        std::find_if(requests.begin(), requests.end(),
          [this, device](const Job& job) {
            return planner_->GetInterpreter()->GetSubgraphIdx(job.model_id, device) != -1;
          });
      if (available_job != requests.end()) {
        Job to_execute = *available_job;
        int subgraph_idx =
          planner_->GetInterpreter()->GetSubgraphIdx(to_execute.model_id, device);
        to_execute.subgraph_idx = subgraph_idx;
        to_execute.device_id = device;
        to_execute.sched_id = planner_->sched_id_++;

        action[device].push_back(to_execute);
        requests.erase(available_job);
        break;
      }
    }
  }
  return action;
}

}  // namespace impl
}  // namespace tflite
