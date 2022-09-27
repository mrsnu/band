#include "band/scheduler/round_robin_scheduler.h"

#include <algorithm>

namespace Band {

ScheduleAction RoundRobinScheduler::Schedule(const Context& context,
                                             JobQueue& requests) {
  ScheduleAction action;
  std::set<WorkerId> idle_workers = context.GetIdleWorkers();

  for (auto worker_id : idle_workers) {
    if (!requests.empty()) {
      auto available_job = std::find_if(
          requests.begin(), requests.end(),
          [this, &context, worker_id](const Job& job) {
            return context.GetModelSubgraphKey(job.model_id, worker_id)
                .IsValid();
          });
      if (available_job != requests.end()) {
        Job to_execute = *available_job;
        SubgraphKey key =
            context.GetModelSubgraphKey(to_execute.model_id, worker_id);
        action[worker_id].push_back({to_execute, key});

        requests.erase(available_job);
      }
    }
  }
  return action;
}

}  // namespace Band
