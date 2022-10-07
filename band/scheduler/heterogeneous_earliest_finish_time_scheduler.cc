#include "band/scheduler/heterogeneous_earliest_finish_time_scheduler.h"

namespace Band {

void HeterogeneousEarliestFinishTimeScheduler::Schedule(
    JobQueue& context.requests_) {
  int window_size =
      std::min(planner_->GetWindowSize(), (int)context.requests_.size());
  // stop if there are no idle devices OR there's nothing in `context.requests_`
  while (window_size > 0) {
    planner_->UpdateWorkerWaitingTime();
    std::set<int> idle_workers = context.GetIdleWorkers();
    if (idle_workers.empty()) {
      break;
    }

    // TODO #172
    // Update HEFT description

    // hold on to a local copy of worker waiting time
    WorkerWaitingTime waiting_time = GetWorkerWaitingTime();

    std::set<int> jobs_to_yield;
    // basically the same as ShortestExpectedLatencyScheduler
    int64_t largest_shortest_latency;
    int target_job_idx;
    int target_subgraph_idx;
    do {
      largest_shortest_latency = -1;
      target_job_idx = -1;
      target_subgraph_idx = -1;

      // only check up to `window_size` context.requests_
      std::set<std::pair<int, int>> searched_jobs;
      for (auto it = context.requests_.begin();
           it != context.requests_.begin() + window_size; ++it) {
        Job& job = *it;

        if (jobs_to_yield.find(job.job_id) != jobs_to_yield.end()) {
          continue;
        }

        std::pair<int, int> job_to_search =
            std::make_pair(job.model_id, job.start_unit_idx);
        if (searched_jobs.find(job_to_search) != searched_jobs.end()) {
          continue;
        } else {
          searched_jobs.insert(job_to_search);
        }

        std::pair<std::vector<int>, int64_t> best_subgraph =
            GetInterpreter()->GetSubgraphWithShortestLatency(job, waiting_time);

        if (largest_shortest_latency < best_subgraph.second) {
          Subgraph* target_subgraph =
              GetInterpreter()->subgraph(best_subgraph.first.front());

          largest_shortest_latency = best_subgraph.second;
          target_subgraph_idx = best_subgraph.first.front();
          target_job_idx = it - context.requests_.begin();
        }
      }

      if (target_job_idx < 0) {
        // no one wants to be scheduled..
        return;
      }

      // skip this job if we can't schedule it immediately,
      // even if this job is the "most urgent" one
      Subgraph* target_subgraph =
          GetInterpreter()->subgraph(target_subgraph_idx);
      int worker_id = target_subgraph->GetKey().worker_id;
      if (idle_workers.find(worker_id) == idle_workers.end()) {
        waiting_time[worker_id] +=
            GetInterpreter()->GetExpectedLatency(target_subgraph_idx);
        auto context.requests__it = context.requests_.begin() + target_job_idx;
        Job job = *context.requests__it;
        jobs_to_yield.insert(job.job_id);
        continue;
      } else {
        break;
      }
    } while (true);

    auto context.requests__it = context.requests_.begin() + target_job_idx;
    Job job = *context.requests__it;

    // erase the job from context.requests_ and decrement window_size
    context.requests_.erase(context.requests__it);
    window_size--;

    Subgraph* target_subgraph = GetInterpreter()->subgraph(target_subgraph_idx);
    // Update Job status specific to this planner.
    // Common status will be updated by `EnqueueAction`.
    if (target_subgraph->IsStart()) {
      // only set these fields if this is the first subgraph of this model
      job.expected_latency = largest_shortest_latency;
    }
    EnqueueAction(job, target_subgraph);
  }
}

}  // namespace Band
}  // namespace Band