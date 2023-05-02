#include "band/scheduler/fixed_worker_scheduler.h"

#include "band/error_reporter.h"

namespace band {
absl::Status FixedWorkerScheduler::Schedule(JobQueue& requests) {
  // TODO: fallback subgraphs for FixedDeviceFixedWorkerPlanner?
  while (!requests.empty()) {
    Job to_execute = requests.front();
    requests.pop_front();  // erase job

    int model_id = to_execute.model_id();

    // Assign worker if not specified.
    WorkerId worker_id = to_execute.HasTargetWorkerId()
                             ? to_execute.target_worker_id()
                             : context_.GetModelWorker(model_id);
    SubgraphKey key = context_.GetLargestSubgraphKey(model_id, worker_id);
    context_.EnqueueToWorker({to_execute, key});
  }
  return absl::OkStatus();
}

}  // namespace band
