#include "tensorflow/lite/planner/least_slack_first_scheduler.h"

#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace impl {

void LeastSlackFirstScheduler::Schedule(JobQueue& requests) {
  int window_size = std::min(planner_->GetWindowSize(), (int)requests.size());
  std::set<int> idle_workers = planner_->GetIdleWorkers();
  if (idle_workers.empty()) {
    // no device is idle; wait for next iteration
    return;
  }

  planner_->UpdateWorkerWaitingTime();
  WorkerWaitingTime waiting_time = GetWorkerWaitingTime();

  for (auto it = requests.begin(); it != requests.begin() + window_size; ++it) {
    Job& job = *it;
    job.expected_latency =
        GetInterpreter()
            ->GetSubgraphWithShortestLatency(job, waiting_time)
            .second;
  }

  // Note Scheduler may fail to enqueue the jobs if they do not
  // have SLOs.
  // TODO: Support jobs with fallback
  int64_t current_time = profiling::time::NowMicros();
  std::sort(requests.begin(), requests.begin() + window_size,
            [&](const Job& first, const Job& second) -> bool {
              return GetSlackTime(current_time, first) <
                     GetSlackTime(current_time, second);
            });

  std::vector<int> indices_to_erase;
  for (auto it = requests.begin(); it != requests.begin() + window_size; ++it) {
    Job job = *it;
    if (GetSlackTime(current_time, job) < 0) {
      indices_to_erase.push_back(it - requests.begin());
      planner_->HandleSLOViolatedJob(job);
      continue;
    }

    std::pair<std::vector<int>, int> best_subgraph =
        GetInterpreter()->GetSubgraphWithShortestLatency(job, waiting_time);
    
    int target_subgraph_idx = best_subgraph.first.front();
    Subgraph* target_subgraph = GetInterpreter()->subgraph(target_subgraph_idx);
    int worker_id = target_subgraph->GetKey().worker_id;
    if (idle_workers.find(worker_id) != idle_workers.end()) {
      indices_to_erase.push_back(it - requests.begin());
      waiting_time[worker_id] +=
          GetInterpreter()->GetExpectedLatency(target_subgraph_idx);
      EnqueueAction(job, target_subgraph);
    }
  }
  for (auto it = indices_to_erase.rbegin(); it != indices_to_erase.rend(); it++) {
    requests.erase(requests.begin() + *it);
  }
}

int64_t LeastSlackFirstScheduler::GetSlackTime(int64_t current_time,
                                               const Job& job) {
  int64_t deadline = job.enqueue_time + job.slo_us;
  int64_t remaining_execution_time = job.expected_latency;
  return deadline - current_time - remaining_execution_time;
}

}  // namespace impl
}  // namespace tflite
