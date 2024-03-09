#include "band/scheduler/dvfs_scheduler.h"

namespace band {

bool DVFSScheduler::Schedule(JobQueue& requests) {
  bool success = true;
  // TODO: fallback subgraphs for FixedDeviceFixedWorkerPlanner?
  while (!requests.empty()) {
    Job to_execute = requests.front();
    requests.pop_front();  // erase job

    int model_id = to_execute.model_id;
    // Priority
    // (1) : direct request from the engine
    // (2) : predefined mapping from the config
    WorkerId worker_id = to_execute.target_worker_id == -1
                             ? engine_.GetModelWorker(model_id)
                             : to_execute.target_worker_id;
    SubgraphKey key = engine_.GetLargestSubgraphKey(model_id, worker_id);
    success &= engine_.EnqueueToWorker({to_execute, key});
  }
  return success;
}

}  // namespace band