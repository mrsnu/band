#include "band/scheduler/round_robin_scheduler.h"

#include <algorithm>

namespace band {

absl::Status RoundRobinScheduler::Schedule(JobQueue& requests) {
  std::set<WorkerId> idle_workers = context_.GetIdleWorkers();

  for (auto worker_id : idle_workers) {
    if (!requests.empty()) {
      auto available_job = std::find_if(
          requests.begin(), requests.end(), [this, worker_id](const Job& job) {
            return context_.GetLargestSubgraphKey(job.model_id(), worker_id)
                .IsValid();
          });
      if (available_job != requests.end()) {
        Job to_execute = *available_job;
        SubgraphKey key =
            context_.GetLargestSubgraphKey(to_execute.model_id(), worker_id);
        context_.EnqueueToWorker({to_execute, key});
        requests.erase(available_job);
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace band
