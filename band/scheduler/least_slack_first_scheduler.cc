#include "band/scheduler/least_slack_first_scheduler.h"

#include <algorithm>

#include "band/time.h"

namespace band {
LeastSlackFirstScheduler::LeastSlackFirstScheduler(Context& context,
                                                   int window_size)
    : IScheduler(context), window_size_(window_size) {}

void LeastSlackFirstScheduler::Schedule(JobQueue& requests) {
  context_.UpdateWorkersWaiting();
  int window_size = std::min(window_size_, (int)requests.size());
  if (window_size <= 0) {
    return;
  }

  std::set<int> idle_workers = context_.GetIdleWorkers();
  if (idle_workers.empty()) {
    return;
  }

  WorkerWaitingTime waiting_time = context_.GetWorkerWaitingTime();

  int64_t current_time = Time::NowMicros();
  SortBySlackTime(requests, window_size, current_time);

  std::set<int> job_indices_to_erase;
  for (auto it = requests.begin(); it != requests.begin() + window_size; ++it) {
    Job job = *it;

    // Get current job's fastest subgraph execution plan + latency
    std::pair<std::vector<SubgraphKey>, int> best_exec_plan =
        context_.GetSubgraphWithShortestLatency(job, waiting_time);
    // Get first executable subgraph plan
    SubgraphKey target_subgraph_key = best_exec_plan.first.front();

    // Change job status and schedule if the execution plan already exceeded SLO
    if (job.slo_us > 0 &&
        current_time + best_exec_plan.second > job.enqueue_time + job.slo_us) {
      job.status = kBandJobSLOViolation;
      context_.EnqueueToWorker({job, target_subgraph_key});
      job_indices_to_erase.insert(it - requests.begin());
      continue;
    }

    // Schedule job if there is a valid idle worker
    int worker_id = target_subgraph_key.GetWorkerId();
    if (idle_workers.find(worker_id) != idle_workers.end()) {
      // Update worker's waiting time as if it will execute the job
      waiting_time[worker_id] += context_.GetExpected(target_subgraph_key);
      context_.EnqueueToWorker({job, target_subgraph_key});
      job_indices_to_erase.insert(it - requests.begin());
      continue;
    }
  }

  for (auto it = job_indices_to_erase.rbegin();
       it != job_indices_to_erase.rend(); ++it) {
    requests.erase(requests.begin() + *it);
  }
}

int64_t LeastSlackFirstScheduler::GetSlackTime(int64_t current_time,
                                               const Job& job) {
  if (job.slo_us > 0) {
    int64_t deadline = job.enqueue_time + job.slo_us;
    int64_t remaining_execution_time = job.expected_latency;
    return deadline - current_time - remaining_execution_time;
  } else {
    return INT_MAX;
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
    it->expected_latency = context_
                               .GetSubgraphWithShortestLatency(
                                   *it, context_.GetWorkerWaitingTime())
                               .second;
  }
}

}  // namespace band