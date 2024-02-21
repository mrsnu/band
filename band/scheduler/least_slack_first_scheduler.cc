#include "band/scheduler/least_slack_first_scheduler.h"

#include <algorithm>

#include "band/device/thermal.h"
#include "band/logger.h"
#include "band/time.h"

namespace band {
LeastSlackFirstScheduler::LeastSlackFirstScheduler(IEngine& engine,
                                                   int window_size)
    : IScheduler(engine), window_size_(window_size) {}

bool LeastSlackFirstScheduler::Schedule(JobQueue& requests) {
  BAND_LOG_PROD(BAND_LOG_INFO, "LeastSlackFirstScheduler::Schedule()");
  bool success = true;
  auto thermal = engine_.GetThermal()->GetAllThermal();
  engine_.UpdateWorkersWaiting();
  int window_size = std::min(window_size_, static_cast<int>(requests.size()));
  if (window_size <= 0) {
    return success;
  }

  std::set<int> idle_workers = engine_.GetIdleWorkers();
  if (idle_workers.empty()) {
    return success;
  }

  WorkerWaitingTime waiting_time = engine_.GetWorkerWaitingTime();

  double current_time = time::NowMicros();
  BAND_LOG_PROD(BAND_LOG_INFO, "Current time: %f", current_time);
  SortBySlackTime(requests, window_size, current_time);

  std::set<int> job_indices_to_erase;
  for (auto it = requests.begin(); it != requests.begin() + window_size; ++it) {
    Job job = *it;

    // Get current job's fastest subgraph execution plan + latency
    std::pair<std::vector<SubgraphKey>, State> best_exec_plan =
        engine_.GetSubgraphWithMinCost(
            job, waiting_time, thermal,
            [](double lat, std::map<SensorFlag, double>) -> double {
              return lat;
            });
    // Get first executable subgraph plan
    SubgraphKey target_subgraph_key = best_exec_plan.first.front();

    // Change job status and schedule if the execution plan already exceeded SLO
    if (job.slo_us > 0 && current_time + std::get<2>(best_exec_plan.second) >
                              job.enqueue_time + job.slo_us) {
      job.status = JobStatus::kSLOViolation;
      success &= engine_.EnqueueToWorker({job, target_subgraph_key});
      job_indices_to_erase.insert(it - requests.begin());
      continue;
    }

    // Schedule job if there is a valid idle worker
    int worker_id = target_subgraph_key.GetWorkerId();
    if (idle_workers.find(worker_id) != idle_workers.end()) {
      // Update worker's waiting time as if it will execute the job
      waiting_time[worker_id] += engine_.GetExpected(target_subgraph_key);
      success &= engine_.EnqueueToWorker({job, target_subgraph_key});
      job_indices_to_erase.insert(it - requests.begin());
      continue;
    }
  }

  for (auto it = job_indices_to_erase.rbegin();
       it != job_indices_to_erase.rend(); ++it) {
    requests.erase(requests.begin() + *it);
  }

  return success;
}

double LeastSlackFirstScheduler::GetSlackTime(double current_time,
                                              const Job& job) {
  if (job.slo_us > 0) {
    double deadline = job.enqueue_time + job.slo_us;
    double remaining_execution_time = job.expected_latency;
    auto slack = deadline - current_time - remaining_execution_time;
    BAND_LOG_PROD(BAND_LOG_INFO, "Job %d (%s) has the slack time %f",
                  job.job_id, job.subgraph_key.ToString().c_str(), slack);
    return slack;
  } else {
    BAND_LOG_PROD(BAND_LOG_WARNING, "Job %d does not have SLO", job.job_id);
    return std::numeric_limits<double>::max() / 2;
  }
}

void LeastSlackFirstScheduler::SortBySlackTime(JobQueue& requests,
                                               int window_size,
                                               double current_time) {
  UpdateExpectedLatency(requests, window_size);
  std::sort(requests.begin(), requests.begin() + window_size,
            [&](const Job& first, const Job& second) -> bool {
              return GetSlackTime(current_time, first) <
                     GetSlackTime(current_time, second);
            });
}

void LeastSlackFirstScheduler::UpdateExpectedLatency(JobQueue& requests,
                                                     int window_size) {
  auto thermal = engine_.GetThermal()->GetAllThermal();
  for (auto it = requests.begin(); it != requests.begin() + window_size; ++it) {
    double expected_lat;
    engine_.GetSubgraphWithMinCost(
        *it, engine_.GetWorkerWaitingTime(),
        thermal,
        [&expected_lat](double lat, std::map<SensorFlag, double>) -> double {
          expected_lat = lat;
          return lat;
        });
    it->expected_latency = expected_lat;
  }
}

}  // namespace band