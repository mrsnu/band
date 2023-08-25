#include "band/scheduler/frame_thermal_scheduler.h"

namespace band {

bool ThermalScheduler::Schedule(JobQueue& requests) {
  bool success = true;
  int num_requests = requests.size();

  while (!requests.empty()) {
    for (auto it = requests.begin(); it != requests.end(); it++) {
      Job& job = *it;
      std::pair<int, BitMask> job_to_search =
          std::make_pair(job.model_id, job.resolved_unit_subgraphs);
      std::set<std::pair<int, BitMask>> searched_jobs;
      if (searched_jobs.find(job_to_search) != searched_jobs.end()) {
        continue;
      } else {
        searched_jobs.insert(job_to_search);
      }

      auto best_subgraph = engine_.GetSubgraphWithMinCost(
        job, 
      )
    }
  }
  return success;
}

}  // namespace band