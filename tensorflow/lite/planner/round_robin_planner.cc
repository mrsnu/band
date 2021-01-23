#include "tensorflow/lite/planner/round_robin_planner.h"

namespace tflite {
namespace impl {

void RoundRobinPlanner::Plan() {
  while (true) {
    if (GetSafeBool().wait())
      return;

    std::vector<bool> is_device_empty;
    for (int i = 0; i < GetInterpreter()->GetWorkersSize(); ++i) {
      Worker* worker = GetInterpreter()->GetWorker(i);
      if (worker) {
        {
          std::lock_guard<std::mutex> lock(worker->GetDeviceMtx());
          bool is_empty = worker->GetDeviceRequests().empty();
          is_device_empty.push_back(is_empty);
        }
      }
    }

    std::deque<Job> local_jobs;
    std::unique_lock<std::mutex> request_lock(GetRequestsMtx());
    if (!GetRequests().empty()) {
      GetRequests().swap(local_jobs);
    } else {
      continue;
    }
    request_lock.unlock();

    while (!local_jobs.empty()) {
      auto empty_device_it =
        std::find(is_device_empty.begin(), is_device_empty.end(), true);
      if (empty_device_it == is_device_empty.end())
        break;
      int device_idx = *(empty_device_it);

      auto available_job =
        std::find_if(local_jobs.begin(), local_jobs.end(),
          [this, device_idx](const Job& job) {
            return GetInterpreter()->GetSubgraphIdx(
                                        job.model_id_, device_idx) != -1;
          });

      if (available_job == local_jobs.end())
        break;

      Job to_execute = *available_job;
      local_jobs.erase(available_job);

      int subgraph_idx =
        GetInterpreter()->GetSubgraphIdx(to_execute.model_id_, device_idx);
      to_execute.subgraph_idx_ = subgraph_idx;
      to_execute.device_id_ = device_idx;

      Worker* worker = GetInterpreter()->GetWorker(to_execute.device_id_);
      if (worker) {
        {
          std::lock_guard<std::mutex> lock(worker->GetDeviceMtx());
          worker->GetDeviceRequests().push_back(to_execute);
          worker->GetRequestCv().notify_one();
        }
        is_device_empty[device_idx] = false;
      }
    }
  }
}

}  // namespace impl
}  // namespace tflite
