#include "tensorflow/lite/planner/round_robin_scheduler.h"

namespace tflite {
namespace impl {

void RoundRobinScheduler::Schedule(JobQueue& requests) {
  std::set<int> idle_workers = planner_->GetIdleWorkers();
  for (auto worker : idle_workers) {
    if (!requests.empty()) {
      auto available_job = std::find_if(
          requests.begin(), requests.end(), [this, worker](const Job& job) {
            return GetInterpreter()->GetSubgraphIdx(job.model_id, worker) != -1;
          });
      if (available_job != requests.end()) {
        Job to_execute = *available_job;
        int subgraph_idx =
            GetInterpreter()->GetSubgraphIdx(to_execute.model_id, worker);
        Subgraph* subgraph = GetInterpreter()->subgraph(subgraph_idx);
        EnqueueAction(to_execute, subgraph);

        requests.erase(available_job);
      }
    }
  }
}

}  // namespace impl
}  // namespace tflite
