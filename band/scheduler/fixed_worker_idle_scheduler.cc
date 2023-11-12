#include "band/scheduler/fixed_worker_idle_scheduler.h"

#include "band/error_reporter.h"

namespace band {
bool FixedWorkerIdleScheduler::Schedule(JobQueue& requests) {
  bool success = true;
  // TODO: fallback subgraphs for FixedDeviceFixedWorkerPlanner?
  while (!requests.empty()) {
    Job to_execute = requests.front();
    requests.pop_front();
    int model_id = to_execute.model_id;

    WorkerId worker_id = to_execute.target_worker_id == -1
                             ? engine_.GetModelWorker(model_id)
                             : to_execute.target_worker_id;
    SubgraphKey key = engine_.GetLargestSubgraphKey(model_id, worker_id);
    success &= engine_.EnqueueToWorker({to_execute, key}, idle_us_);
  }
  return success;
}

}  // namespace band
