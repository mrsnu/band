#include "band/scheduler/round_robin_idle_scheduler.h"

#include <algorithm>

#include "band/logger.h"

namespace band {

bool RoundRobinIdleScheduler::Schedule(JobQueue& requests) {
  BAND_LOG_PROD(BAND_LOG_INFO, "RoundRobinIdleScheduler::Schedule");
  std::set<WorkerId> idle_workers = engine_.GetIdleWorkers();
  bool success = true;

  for (auto worker_id : idle_workers) {
    if (!requests.empty()) {
      auto available_job = std::find_if(
          requests.begin(), requests.end(), [this, worker_id](const Job& job) {
            return engine_.GetLargestSubgraphKey(job.model_id, worker_id)
                .IsValid();
          });
      if (available_job != requests.end()) {
        Job to_execute = *available_job;
        SubgraphKey key =
            engine_.GetLargestSubgraphKey(to_execute.model_id, worker_id);
        success &= engine_.EnqueueToWorker(std::make_pair(to_execute, key), idle_us_);
        requests.erase(available_job);
      }
    }
  }

  return success;
}

}  // namespace band
