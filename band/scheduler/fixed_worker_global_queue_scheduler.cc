#include "band/error_reporter.h"
#include "band/scheduler/fixed_worker_scheduler.h"
#include "band/time.h"

namespace band {

bool FixedWorkerGlobalQueueScheduler::Schedule(JobQueue& requests) {
  bool success = true;
  // TODO: fallback subgraphs for FixedDeviceFixedWorkerPlanner?
  std::set<int> idle_workers = context_.GetIdleWorkers();
  if (idle_workers.empty()) {
    // no device is idle; wait for next iteration
    return success;
  }

  BAND_REPORT_ERROR(DefaultErrorReporter(), "NOT IMPLEMENTED");
  return success;
  // for (auto it = context_.requests_.begin(); it !=
  // context_.requests_.end();)
  // {
  //   Job &to_execute = *it;
  //   int model_id = subgraph_key.GetModelId();

  //   int worker_id;
  //   if (worker_id >= 0 && GetModelExecutor()->GetNumWorkers()) {
  //     worker_id = to_execute.worker_id;
  //   } else {
  //     worker_id = planner_->GetModelWorkerMap()[model_id];
  //   }

  //   auto idle_workers_it = idle_workers.find(worker_id);
  //   if (idle_workers_it == idle_workers.end()) {
  //     // that device is not idle, so leave this job alone for now
  //     ++it;
  //     continue;
  //   }

  //   int subgraph_idx = context_.GetSubgraphIdx(model_id, worker_id);
  //   Subgraph *subgraph = GetModelExecutor()->subgraph(subgraph_idx);
  //   to_execute.expected_latency =
  //       GetWorkerWaitingTime()[worker_id] +
  //       GetModelExecutor()->GetExpectedLatency(subgraph_idx);
  //   EnqueueAction(to_execute, subgraph);

  //   // delete this job from our request queue and
  //   // delete this device from our idle_workers set
  //   it = context_.requests_.erase(it);
  //   idle_workers.erase(idle_workers_it);

  //   if (idle_workers.empty()) {
  //     // no device is idle; wait for next iteration
  //     break;
  //   }
  // }
}

}  // namespace band
