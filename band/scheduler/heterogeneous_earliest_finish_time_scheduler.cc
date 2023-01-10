#include "band/scheduler/heterogeneous_earliest_finish_time_scheduler.h"

#include <unordered_set>

namespace Band {

ScheduleAction HEFTScheduler::Schedule(const Context& context,
                                       JobQueue& requests) {
  ScheduleAction action;
  int window_size = std::min(window_size, (int)requests.size());

  // stop if there are no idle devices OR there's nothing in `requests`
  while (window_size > 0) {
    std::set<int> idle_workers = context.GetIdleWorkers();
    if (idle_workers.empty()) {
      break;
    }

    // TODO #172
    // Update HEFT description

    // hold on to a local copy of worker waiting time
    WorkerWaitingTime waiting_time = context.GetWorkerWaitingTime();

    std::set<JobId> jobs_to_yield;
    // basically the same as ShortestExpectedLatencyScheduler
    int64_t largest_shortest_latency;
    JobId target_job_id;
    SubgraphKey target_subgraph_key;
    do {
      largest_shortest_latency = -1;
      target_job_id = -1;

      // only check up to `window_size` requests
      std::unordered_set<std::pair<int, BitMask>, CacheHash> searched_jobs;
      for (auto it = requests.begin(); it != requests.begin() + window_size;
           ++it) {
        Job& job = *it;

        if (jobs_to_yield.find(job.job_id) != jobs_to_yield.end()) {
          continue;
        }

        std::pair<int, BitMask> job_to_search =
            std::make_pair(job.model_id, job.resolved_unit_subgraphs);
        if (searched_jobs.find(job_to_search) != searched_jobs.end()) {
          continue;
        } else {
          searched_jobs.insert(job_to_search);
        }

        std::pair<std::vector<SubgraphKey>, int64_t> best_subgraph =
            context.GetSubgraphWithShortestLatency(job, waiting_time);

        if (largest_shortest_latency < best_subgraph.second) {
          largest_shortest_latency = best_subgraph.second;
          target_subgraph_key = best_subgraph.first.front();
          target_job_id = it - requests.begin();
        }
      }

      if (target_job_id < 0) {
        // no one wants to be scheduled..
        return;
      }

      // skip this job if we can't schedule it immediately,
      // even if this job is the "most urgent" one
      const int worker_id = target_subgraph_key.GetWorkerId();
      if (idle_workers.find(worker_id) == idle_workers.end()) {
        waiting_time[worker_id] += context.GetExpected(target_subgraph_key);
        auto requests_it = requests.begin() + target_job_id;
        Job job = *requests_it;
        jobs_to_yield.insert(job.job_id);
        continue;
      } else {
        break;
      }
    } while (true);

    auto requests_it = requests.begin() + target_job_id;
    Job job = *requests_it;

    // erase the job from context.requests_ and decrement window_size
    requests.erase(requests_it);
    window_size--;

    // Update Job status specific to this planner.
    // Common status will be updated by `EnqueueAction`.
    if (context.IsBegin(target_subgraph_key)) {
      // only set these fields if this is the first subgraph of this model
      job.expected_latency = largest_shortest_latency;
    }
    action[target_subgraph_key.GetWorkerId()].push_back(
        {job, target_subgraph_key});
  }
}

}  // namespace Band
