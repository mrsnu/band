#include "band/scheduler/least_slack_first_scheduler.h"

#include <algorithm>

#include "band/time.h"

namespace Band {
LeastSlackFirstScheduler::LeastSlackFirstScheduler(int window_size)
    : window_size_(window_size) {}

ScheduleAction LeastSlackFirstScheduler::Schedule(const Context& context) {
  int window_size = std::min(window_size_, (int)context.requests_.size());
  if (window_size <= 0) {
    return;
  }

  std::set<int> idle_workers = context.GetIdleWorkers();
  if (idle_workers.empty()) {
    return;
  }

  context.UpdateWorkerWaitingTime();
  WorkerWaitingTime waiting_time = context.GetWorkerWaitingTime();

  int64_t current_time = Time::NowMicros();
  JobQueue current_requests = context.requests_;
  SortBySlackTime(current_requests, window_size, current_time);

  std::set<int> job_indices_to_erase;
  for (auto it = current_requests.begin();
       it != current_requests.begin() + window_size; ++it) {
    Job job = *it;

    std::pair<std::vector<int>, int> best_subgraph =
        GetInterpreter()->GetSubgraphWithShortestLatency(job, waiting_time);

    int target_subgraph_idx = best_subgraph.first.front();
    Subgraph* target_subgraph = GetInterpreter()->subgraph(target_subgraph_idx);
    if (job.slo_us > 0 &&
        current_time + best_subgraph.second > job.enqueue_time + job.slo_us) {
      job.status = kBandJobSLOViolation;
      EnqueueAction(job, target_subgraph);
      job_indices_to_erase.insert(it - current_requests.begin());
      continue;
    }

    int worker_id = target_subgraph->GetKey().worker_id;
    if (idle_workers.find(worker_id) != idle_workers.end()) {
      waiting_time[worker_id] +=
          GetInterpreter()->GetExpectedLatency(target_subgraph_idx);
      EnqueueAction(job, target_subgraph);
      job_indices_to_erase.insert(it - current_requests.begin());
      continue;
    }
  }
  for (auto it = job_indices_to_erase.rbegin();
       it != job_indices_to_erase.rend(); ++it) {
    current_requests.erase(current_requests.begin() + *it);
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

void LeastSlackFirstScheduler::SortBySlackTime(const Context& context,
                                               JobQueue& requests,
                                               int window_size,
                                               int64_t current_time) {
  UpdateExpectedLatency(context, requests, window_size);
  std::sort(requests.begin(), requests.begin() + window_size,
            [&](const Job& first, const Job& second) -> bool {
              return GetSlackTime(current_time, first) <
                     GetSlackTime(current_time, second);
            });
}

void LeastSlackFirstScheduler::UpdateExpectedLatency(const Context& context,
                                                     JobQueue& requests,
                                                     int window_size) {
  for (auto it = requests.begin(); it != requests.begin() + window_size; ++it) {
    it->expected_latency =
        context
            .GetSubgraphWithShortestLatency(*it, context.GetWorkerWaitingTime())
            .second;
  }
}

}  // namespace Band