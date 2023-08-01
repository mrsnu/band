#include "band/scheduler/least_slack_first_scheduler.h"

#include <algorithm>

#include "band/time.h"
#include "band/logger.h"

namespace band {
LeastSlackFirstScheduler::LeastSlackFirstScheduler(IEngine& engine,
                                                   int window_size)
    : IScheduler(engine), window_size_(window_size) {}

bool LeastSlackFirstScheduler::Schedule(JobQueue& requests) {
  bool success = true;
  engine_.UpdateWorkersWaiting();
  int window_size = std::min(window_size_, (int)requests.size());
  if (window_size <= 0) {
    return success;
  }

  std::set<int> idle_workers = engine_.GetIdleWorkers();
  if (idle_workers.empty()) {
    return success;
  }

  WorkerWaitingTime waiting_time = engine_.GetWorkerWaitingTime();

  int64_t current_time = time::NowMicros();
  SortBySlackTime(requests, window_size, current_time);

  std::set<int> job_indices_to_erase;
  for (auto it = requests.begin(); it != requests.begin() + window_size; ++it) {
    Job job = *it;

    // Get current job's fastest subgraph execution plan + latency
    std::pair<std::vector<SubgraphKey>, int> best_exec_plan =
        engine_.GetSubgraphWithShortestLatency(job, waiting_time);
    // Get first executable subgraph plan
    SubgraphKey target_subgraph_key = best_exec_plan.first.front();

    // Change job status and schedule if the execution plan already exceeded SLO
    if (job.HasSLO() &&
        current_time + best_exec_plan.second > job.enqueue_time + job.slo_us()) {
      auto status = job.SLOViolation();
      if (!status.ok()) {
        success = false;
      }
      success &= engine_.EnqueueToWorker({job, target_subgraph_key});
      job_indices_to_erase.insert(it - requests.begin());
      continue;
    }

    // Schedule job if there is a valid idle worker
    int worker_id = target_subgraph_key.GetWorkerId();
    if (idle_workers.find(worker_id) != idle_workers.end()) {
      // Update worker's waiting time as if it will execute the job
      auto status_or_expected_time = engine_.GetExpected(target_subgraph_key);
      if (!status_or_expected_time.ok()) {
        BAND_LOG_PROD(BAND_LOG_WARNING,
                      "Failed to get expected latency for subgraph key %s",
                      target_subgraph_key.ToString().c_str());
        return false;
      }
      int64_t expected_time = status_or_expected_time.value();
      waiting_time[worker_id] += expected_time;
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

int64_t LeastSlackFirstScheduler::GetSlackTime(int64_t current_time,
                                               const Job& job) {
  if (job.HasSLO()) {
    int64_t deadline = job.enqueue_time + job.slo_us();
    int64_t remaining_execution_time = job.expected_latency;
    return deadline - current_time - remaining_execution_time;
  } else {
    return std::numeric_limits<int>::max();
  }
}

void LeastSlackFirstScheduler::SortBySlackTime(JobQueue& requests,
                                               int window_size,
                                               int64_t current_time) {
  UpdateExpectedLatency(requests, window_size);
  std::sort(requests.begin(), requests.begin() + window_size,
            [&](const Job& first, const Job& second) -> bool {
              return GetSlackTime(current_time, first) <
                     GetSlackTime(current_time, second);
            });
}

void LeastSlackFirstScheduler::UpdateExpectedLatency(JobQueue& requests,
                                                     int window_size) {
  for (auto it = requests.begin(); it != requests.begin() + window_size; ++it) {
    it->expected_latency = engine_
                               .GetSubgraphWithShortestLatency(
                                   *it, engine_.GetWorkerWaitingTime())
                               .second;
  }
}

}  // namespace band