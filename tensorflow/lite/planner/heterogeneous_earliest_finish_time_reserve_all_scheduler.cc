#include "tensorflow/lite/planner/heterogeneous_earliest_finish_time_reserve_all_scheduler.h"

namespace tflite {
namespace impl {

void HeterogeneousEarliestFinishTimeReserveAllScheduler::Schedule(JobQueue& requests) {
  int window_size = std::min(planner_->GetWindowSize(), (int)requests.size());
  planner_->UpdateWorkerWaitingTime();
  WorkerWaitingTime waiting_time = GetWorkerWaitingTime();
  std::set<int> jobs_to_yield;
  ReservedTime reserved_time;

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
    ScheduleLog log;

    // only check up to `window_size` requests
    for (auto it = requests.begin(); it != requests.begin() + window_size;
         ++it) {
      Job& job = *it;

      if (jobs_to_yield.find(job.job_id) != jobs_to_yield.end()) {
        continue;
      }

      std::pair<std::vector<int>, int64_t> best_subgraph =
          GetInterpreter()->GetSubgraphWithShortestLatency(job, waiting_time,
                                                           reserved_time);

      planner_->LogScheduleStep(log, job.model_id, job.start_unit_idx,
                                best_subgraph.second);

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

    planner_->LogSchedule(waiting_time, target_subgraphs, log);

    // skip this job if we can't schedule it immediately,
    // even if this job is the "most urgent" one
    const int target_subgraph_idx = target_subgraphs.front();
    Subgraph* target_subgraph = GetInterpreter()->subgraph(target_subgraph_idx);
    const int worker_id = target_subgraph->GetKey().worker_id;
    if (idle_workers.find(worker_id) == idle_workers.end()) {
      auto requests_it = requests.begin() + target_job_idx;
      jobs_to_yield.insert(requests_it->job_id);

      int64_t prev_end_time = waiting_time[worker_id];
      prev_end_time = prev_end_time < 0 ? 0 : prev_end_time;
      for (int i = 0; i < target_subgraphs.size(); i++) {
        const int subgraph_idx = target_subgraphs[i];
        Subgraph* subgraph = GetInterpreter()->subgraph(subgraph_idx);
        const int wid = subgraph->GetKey().worker_id;
        const int exe_time = GetInterpreter()->GetExpectedLatency(subgraph_idx);
        for (const auto& start_end : reserved_time[wid]) {
          if (prev_end_time + exe_time <= start_end.first) {
            break;
          }
          prev_end_time = start_end.second;
        }
        reserved_time[wid].insert(
            std::make_pair(prev_end_time, prev_end_time + exe_time));
      }
    } else {
      waiting_time[worker_id] +=
          GetInterpreter()->GetExpectedLatency(target_subgraph_idx);
      auto requests_it = requests.begin() + target_job_idx;
      Job job = *requests_it;

      // erase the job from requests and decrement window_size
      requests.erase(requests_it);
      window_size--;

      // Update Job status specific to this planner.
      // Common status will be updated by `EnqueueAction`.
      if (target_subgraph->IsStart()) {
        // only set these fields if this is the first subgraph of this model
        job.expected_latency = largest_shortest_latency;
      }
      EnqueueAction(job, target_subgraph);

      // target_subgraphs.front() is already enqueued, so start from 1.
      int64_t prev_end_time = waiting_time[worker_id];
      prev_end_time = prev_end_time < 0 ? 0 : prev_end_time;
      for (int i = 1; i < target_subgraphs.size(); i++) {
        const int subgraph_idx = target_subgraphs[i];
        Subgraph* subgraph = GetInterpreter()->subgraph(subgraph_idx);
        const int wid = subgraph->GetKey().worker_id;
        const int exe_time = GetInterpreter()->GetExpectedLatency(subgraph_idx);
        for (const auto& start_end : reserved_time[wid]) {
          if (prev_end_time + exe_time <= start_end.first) {
            break;
          }
          prev_end_time = start_end.second;
        }
        reserved_time[wid].insert(
            std::make_pair(prev_end_time, prev_end_time + exe_time));
      }
    }
  }
}

}  // namespace impl
}  // namespace tflite
