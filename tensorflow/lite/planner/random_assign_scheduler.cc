#include "tensorflow/lite/planner/random_assign_scheduler.h"
#include <random>

namespace tflite {
namespace impl {

void RandomAssignScheduler::Schedule(JobQueue& requests) {
  std::set<int> idle_workers = planner_->GetIdleWorkers();
  auto worker_id = rand() % idle_workers.size();
  if (!requests.empty()) {
    auto available_job = std::find_if(
        requests.begin(), requests.end(), [this, worker_id](const Job& job) {
          return GetInterpreter()->GetSubgraphIdx(job.model_id, worker_id) != -1;
        });
    if (available_job != requests.end()) {
      Job to_execute = *available_job;
      int subgraph_idx =
          GetInterpreter()->GetSubgraphIdx(to_execute.model_id, worker_id);
      Subgraph* subgraph = GetInterpreter()->subgraph(subgraph_idx);
      EnqueueAction(to_execute, subgraph);

      requests.erase(available_job);
    }
  }
}

}  // namespace impl
}  // namespace tflite
