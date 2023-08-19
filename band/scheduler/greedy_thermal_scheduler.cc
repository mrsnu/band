#include "band/scheduler/greedy_thermal_scheduler.h"

namespace band {

bool GreedyThremalScheduler::Schedule(JobQueue& requests) {
  bool success = true;

  while (!requests.empty()) {
    engine_.UpdateWorkersWaiting();
    std::set<int> idle_workers = engine_.GetIdleWorkers();
    if (idle_workers.empty()) {
      break;
    }

    Job job = requests.front();
    requests.pop_front();
  }

  return success;
}

std::pair<int, double> GetMinCostSubgraph(Job& job,
                                          WorkerWaitingTime& waiting_time) {
  double min_slo_cost = std::numeric_limits<double>::max();
  int min_index = -1;
  return {};
}

}  // namespace band