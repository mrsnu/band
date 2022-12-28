#include "band/scheduler/fixed_worker_scheduler.h"

#include "band/error_reporter.h"

namespace Band {
ScheduleAction FixedWorkerScheduler::Schedule(const Context& context,
                                              JobQueue& requests) {
  ScheduleAction action;

  // TODO: fallback subgraphs for FixedDevicePlanner?
  while (!requests.empty()) {
    Job to_execute = requests.front();
    requests.pop_front();  // erase job

    int model_id = to_execute.model_id;
    // Priority
    // (1) : direct request from the engine
    // (2) : predefined mapping from the config
    WorkerId worker_id = to_execute.target_worker_id == -1
                             ? context.GetModelWorker(model_id)
                             : to_execute.target_worker_id;
    SubgraphKey key = context.GetLargestSubgraphKey(model_id, worker_id);
    action[worker_id].push_back({to_execute, key});
  }
  return action;
}

}  // namespace Band
