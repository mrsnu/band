#include "band/scheduler/greedy_thermal_scheduler.h"

namespace band {

bool GreedyThremalScheduler::Schedule(JobQueue& requests) {
  bool success = true;
  int num_requests = requests.size();

  engine_.UpdateWorkersWaiting();
  std::set<int> idle_workers = engine_.GetIdleWorkers();
  if (idle_workers.empty()) {
    return success;
  }

  WorkerWaitingTime waiting_time = engine_.GetWorkerWaitingTime();
  
  
  return success;
}

}  // namespace band