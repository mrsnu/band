#include "band/scheduler/fixed_worker_scheduler.h"

#include "band/error_reporter.h"

namespace band {
void FixedWorkerScheduler::Schedule(JobQueue& requests) {
  // TODO: fallback subgraphs for FixedDeviceFixedWorkerPlanner?
  while (!requests.empty()) {
    Job to_execute = requests.front();
    requests.pop_front();  // erase job

    int model_id = to_execute.model_id;
    // Priority
    // (1) : direct request from the engine
    // (2) : predefined mapping from the config
    WorkerId worker_id = to_execute.target_worker_id == -1
                             ? context_.GetModelWorker(model_id)
                             : to_execute.target_worker_id;
    SubgraphKey key = context_.GetLargestSubgraphKey(model_id, worker_id);
    context_.EnqueueToWorker({to_execute, key});
  }
}

}  // namespace band
