#include "band/scheduler/thermal_scheduler.h"

#include <unordered_set>

#include "band/job_tracer.h"
#include "band/logger.h"
#include "band/time.h"

namespace band {

bool ThermalScheduler::Schedule(JobQueue& requests) {
<<<<<<< HEAD
  // BAND_TRACER_SCOPED_THREAD_EVENT(Schedule);
=======
  BAND_TRACER_SCOPED_THREAD_EVENT(Schedule);
>>>>>>> 7c3858e6... (temporal) Add scoped tracing
  bool success = true;
  int num_requests = requests.size();

  while (!requests.empty()) {
    engine_.UpdateWorkersWaiting();
    WorkerWaitingTime worker_waiting = engine_.GetWorkerWaitingTime();
    double largest_min_cost = -1;
    double largest_expected_latency = -1;
    std::map<SensorFlag, double> largest_expected_thermal;
    int target_job_idx;
    SubgraphKey target_subgraph_key;

    for (auto it = requests.begin(); it != requests.end(); it++) {
      Job& job = *it;
      std::pair<int, BitMask> job_to_search =
          std::make_pair(job.model_id, job.resolved_unit_subgraphs);
      std::unordered_set<std::pair<int, BitMask>, JobIdBitMaskHash>
          searched_jobs;
      if (searched_jobs.find(job_to_search) != searched_jobs.end()) {
        continue;
      } else {
        searched_jobs.insert(job_to_search);
      }

      double expected_lat;
      std::map<SensorFlag, double> expected_therm;
      auto best_subgraph = engine_.GetSubgraphWithMinCost(
          job, worker_waiting,
          [&expected_lat, &expected_therm](
              double lat, const std::map<SensorFlag, double>& therm) -> double {
            return therm.at(SensorFlag::kTarget);
          });

      if (largest_min_cost < best_subgraph.second) {
        largest_min_cost = best_subgraph.second;
        largest_expected_latency = expected_lat;
        largest_expected_thermal = expected_therm;
        target_job_idx = it - requests.begin();
        target_subgraph_key = best_subgraph.first[0];
      }
    }

    if (target_subgraph_key.IsValid() == false) {
      continue;
    }

    Job most_urgent_job = requests[target_job_idx];
    most_urgent_job.expected_latency = largest_expected_latency;
    most_urgent_job.expected_thermal = largest_expected_thermal;
    BAND_LOG_PROD(BAND_LOG_INFO, "Expected thermal:");
    for (auto& pair : largest_expected_thermal) {
      BAND_LOG_PROD(BAND_LOG_INFO, "  %s: %f", ToString(pair.first),
                    pair.second);
    }

    requests.erase(requests.begin() + target_job_idx);
    success &= engine_.EnqueueToWorker({most_urgent_job, target_subgraph_key});
  }
  return success;
}

}  // namespace band