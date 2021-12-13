#include "tensorflow/lite/planner/heterogeneous_earliest_finish_time_scheduler.h"

namespace tflite {
namespace impl {

void HeterogeneousEarliestFinishTimeScheduler::Schedule(JobQueue& requests) {
  int window_size = std::min(planner_->GetWindowSize(), (int)requests.size());
  planner_->UpdateWorkerWaitingTime();
  WorkerWaitingTime waiting_time = GetWorkerWaitingTime();
  std::set<int> jobs_to_yield;

  // stop if there are no idle devices OR there's nothing in `requests`
  while (window_size > jobs_to_yield.size()) {
    std::set<int> idle_workers;
    for (int wid = 0; wid < GetInterpreter()->GetNumWorkers(); wid++) {
      if (waiting_time[wid] == 0) {
        idle_workers.insert(wid);
      }
    }
    if (idle_workers.empty()) {
      break;
    }

    // basically the same as ShortestExpectedLatencyScheduler
    int64_t largest_shortest_latency = -1;
    int target_job_idx = -1;
    std::vector<int> target_subgraphs;

    // only check up to `window_size` requests
    for (auto it = requests.begin(); it != requests.begin() + window_size; ++it) {
      Job& job = *it;

      if (jobs_to_yield.find(job.job_id) != jobs_to_yield.end()) {
        continue;
      }

      std::pair<std::vector<int>, int64_t> best_subgraph =
          GetInterpreter()->GetSubgraphWithShortestLatency(job, waiting_time);

      if (largest_shortest_latency < best_subgraph.second) {
        Subgraph* target_subgraph =
            GetInterpreter()->subgraph(best_subgraph.first.front());

        largest_shortest_latency = best_subgraph.second;
        target_subgraphs = best_subgraph.first;
        target_job_idx = it - requests.begin();
      }
    }

    if (target_job_idx < 0) {
      // no one wants to be scheduled..
      return;
    }

    // skip this job if we can't schedule it immediately,
    // even if this job is the "most urgent" one
    const int& target_subgraph_idx = target_subgraphs.front();
    Subgraph* target_subgraph = GetInterpreter()->subgraph(target_subgraph_idx);
    int worker_id = target_subgraph->GetKey().worker_id;
    waiting_time[worker_id] +=
        GetInterpreter()->GetExpectedLatency(target_subgraph_idx);
    if (idle_workers.find(worker_id) == idle_workers.end()) {
      auto requests_it = requests.begin() + target_job_idx;
      jobs_to_yield.insert(requests_it->job_id);
    } else {
      auto requests_it = requests.begin() + target_job_idx;
      Job job = *requests_it;

      // erase the job from requests and decrement window_size
      requests.erase(requests_it);
      window_size--;

      Subgraph* target_subgraph =
          GetInterpreter()->subgraph(target_subgraph_idx);
      // Update Job status specific to this planner.
      // Common status will be updated by `EnqueueAction`.
      if (target_subgraph->IsStart()) {
        // only set these fields if this is the first subgraph of this model
        job.expected_latency = largest_shortest_latency;
      }
      EnqueueAction(job, target_subgraph);
    }
  }
}

}  // namespace impl
}  // namespace tflite
