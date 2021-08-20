#include "tensorflow/lite/planner/least_slack_first_scheduler.h"

#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace impl {

void LeastSlackFirstScheduler::Schedule(JobQueue& requests) {
  // Note Scheduler may fail to enqueue the jobs if they do not
  // have SLOs.
  // TODO: Support jobs with fallback
  SortBySlackTime(requests);

  planner_->UpdateWorkerWaitingTime();
  // hold on to a local copy of worker waiting time
  WorkerWaitingTime waiting_time = GetWorkerWaitingTime();
  for (auto it = requests.begin(); it != requests.end();) {
    std::set<int> idle_workers = planner_->GetIdleWorkers();
    if (idle_workers.empty()) {
      // no device is idle; wait for next iteration
      return;
    }
    Job& next_job = *it;
    int best_subgraph_idx =
        GetInterpreter()->GetSubgraphIdxSatisfyingSLO(next_job, waiting_time, idle_workers);

    // If the target device is not idle, give opportunity to the next job.
    if (best_subgraph_idx == -1) {
      ++it;
      continue;
    }

    Subgraph* target_subgraph = GetInterpreter()->subgraph(best_subgraph_idx);

    EnqueueAction(next_job, target_subgraph);
    it = requests.erase(it);

    planner_->UpdateWorkerWaitingTime();
    waiting_time = GetWorkerWaitingTime();
  }
}

int64_t LeastSlackFirstScheduler::GetSlackTime(int64_t current_time,
                                               const Job& job) {
  int64_t deadline = job.enqueue_time + job.slo_us;
  int64_t remaining_execution_time = job.expected_latency;
  return deadline - current_time - remaining_execution_time;
}

void LeastSlackFirstScheduler::SortBySlackTime(JobQueue& requests) {
  UpdateExpectedLatency(requests);
  int64_t current_time = profiling::time::NowMicros();
  std::sort(requests.begin(), requests.end(),
            [&](const Job& first, const Job& second) -> bool {
              return GetSlackTime(current_time, first) <
                     GetSlackTime(current_time, second);
            });
}

void LeastSlackFirstScheduler::UpdateExpectedLatency(JobQueue& requests) {
  for (auto& request : requests) {
    request.expected_latency =
        GetInterpreter()->GetSubgraphWithShortestLatency(request, GetWorkerWaitingTime()).second;
  }
}

}  // namespace impl
}  // namespace tflite
