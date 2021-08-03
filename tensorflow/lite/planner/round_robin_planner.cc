#include "tensorflow/lite/planner/round_robin_planner.h"

namespace tflite {
namespace impl {

void RoundRobinPlanner::Plan() {
  while (true) {
    if (GetSafeBool().wait())
      return;

    std::vector<bool> is_device_empty;
    for (int i = 0; i < kTfLiteNumDevices; ++i) {
      Worker* worker = GetInterpreter()->GetWorker(i);
      if (!worker || !worker->IsAvailable()) {
        continue;
      }

      {
        std::lock_guard<std::mutex> lock(worker->GetDeviceMtx());
        bool is_empty = worker->GetDeviceRequests().empty();
        is_device_empty.push_back(is_empty);
      }
    }

    std::unique_lock<std::mutex> request_lock(GetRequestsMtx());
    while (!GetRequests().empty()) {
      int device_idx;
      for (device_idx = is_device_empty.size() - 1; device_idx >= 0; --device_idx) {
        if (is_device_empty[device_idx]) {
          auto available_job =
            std::find_if(GetRequests().begin(), GetRequests().end(),
              [this, device_idx](const Job& job) {
                return GetInterpreter()->GetSubgraphIdx(
                                            job.model_id, device_idx) != -1;
              });
          if (available_job != GetRequests().end()) {
            Job to_execute = *available_job;
            int subgraph_idx = GetInterpreter()->GetSubgraphIdx(to_execute.model_id, device_idx);
            SubgraphKey& key = GetInterpreter()->subgraph(subgraph_idx)->GetKey();
            UpdateJobEnqueueStatus(to_execute, key);

            Worker* worker = GetInterpreter()->GetWorker(to_execute.device_id);
            if (worker->GiveJob(to_execute)) {
              // all is well
              // delete this job from our request queue
              GetRequests().erase(available_job);
              sched_id_++;
            }
            is_device_empty[device_idx] = false;
            break;
          }
        }
      }

      if (device_idx < 0)
        break;
    }
    request_lock.unlock();
  }
}

bool RoundRobinPlanner::NeedProfile() {
  return false;
}

}  // namespace impl
}  // namespace tflite
