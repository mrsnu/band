#include "band/scheduler/fixed_worker_scheduler.h"

#include "band/error_reporter.h"
#include "band/logger.h"

namespace band {

bool FixedWorkerScheduler::Schedule(JobQueue& requests) {
  bool success = true;
  // TODO: fallback subgraphs for FixedDeviceFixedWorkerPlanner?
  while (!requests.empty()) {
    Job to_execute = requests.front();
    requests.pop_front();  // erase job

    int model_id = to_execute.model_id();

    if (!to_execute.HasTargetWorker()) {
      BAND_LOG_PROD(BAND_LOG_ERROR, "Job does not have target worker.");
      return false;
    }
    WorkerId worker_id = to_execute.target_worker_id();
    SubgraphKey key = engine_.GetLargestSubgraphKey(model_id, worker_id);
    success &= engine_.EnqueueToWorker({to_execute, key});
  }
  return success;
}

}  // namespace band
