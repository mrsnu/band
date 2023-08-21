#include "band/scheduler/frame_thermal_scheduler.h"

namespace band {

bool FrameThermalScheduler::Schedule(JobQueue& requests) {
  bool success = true;
  int num_requests = requests.size();

  engine_.UpdateWorkersWaiting();
  std::set<int> idle_workers = engine_.GetIdleWorkers();
  if (idle_workers.empty()) {
    return success;
  }

  WorkerWaitingTime waiting_time = engine_.GetWorkerWaitingTime();
  std::set<JobId> jobs_to_yield;

  double largest_shortest_latency;
  int target_job_index;
  SubgraphKey target_subgraph_key;
  SubgraphKey target_subgraph_key_next;
  
  return success;
}

}  // namespace band