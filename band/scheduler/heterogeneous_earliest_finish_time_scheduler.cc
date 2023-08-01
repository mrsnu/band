#include "band/scheduler/heterogeneous_earliest_finish_time_scheduler.h"

#include <unordered_set>

#include "band/logger.h"

namespace band {
HEFTScheduler::HEFTScheduler(IEngine& engine, int window_size, bool reserve)
    : IScheduler(engine), window_size_(window_size), reserve_(reserve) {}

bool HEFTScheduler::Schedule(JobQueue& requests) {
  bool success = true;
  int window_size = std::min(window_size_, (int)requests.size());
  // stop if there are no idle devices OR there's nothing in `requests`
  while (window_size > 0) {
    engine_.UpdateWorkersWaiting();
    std::set<int> idle_workers = engine_.GetIdleWorkers();
    if (idle_workers.empty()) {
      break;
    }

    // hold on to a local copy of worker waiting time
    WorkerWaitingTime waiting_time = engine_.GetWorkerWaitingTime();
    std::set<JobId> jobs_to_yield;
    // basically the same as ShortestExpectedLatencyScheduler
    int64_t largest_shortest_latency;
    int64_t target_job_index;
    SubgraphKey target_subgraph_key;
    SubgraphKey target_subgraph_key_next;
    do {
      largest_shortest_latency = -1;
      target_job_index = -1;

      // only check up to `window_size` requests
      std::unordered_set<std::pair<int, BitMask>, JobIdBitMaskHash>
          searched_jobs;
      for (auto it = requests.begin(); it != requests.begin() + window_size;
           ++it) {
        Job job = *it;

        if (jobs_to_yield.find(job.id()) != jobs_to_yield.end()) {
          continue;
        }

        std::pair<int, BitMask> job_to_search =
            std::make_pair(job.model_id(), job.resolved_unit_subgraphs);
        if (searched_jobs.find(job_to_search) != searched_jobs.end()) {
          continue;
        } else {
          searched_jobs.insert(job_to_search);
        }

        // update waiting_time for all future jobs in reserved_
        WorkerWaitingTime reserved_time(waiting_time);
        for (auto job_subgraph_key : reserved_) {
          if (job_subgraph_key.first == job.id()) {
            continue;
          }
          auto maybe_expected_time =
              engine_.GetExpected(job_subgraph_key.second);
          if (!maybe_expected_time.has_value()) {
            BAND_LOG_PROD(BAND_LOG_ERROR,
                          "Failed to get expected latency for subgraph key %s",
                          job_subgraph_key.second.ToString().c_str());
          }
          int64_t expected_time = maybe_expected_time.value();
          reserved_time[job_subgraph_key.second.GetWorkerId()] += expected_time;
        }

        std::pair<std::vector<SubgraphKey>, int64_t> best_subgraph =
            engine_.GetSubgraphWithShortestLatency(job, reserved_time).value();

        if (largest_shortest_latency < best_subgraph.second) {
          largest_shortest_latency = best_subgraph.second;
          target_subgraph_key = best_subgraph.first.front();
          target_job_index = it - requests.begin();
          if (best_subgraph.first.size() > 1) {
            target_subgraph_key_next = best_subgraph.first[1];
          } else {
            target_subgraph_key_next = {};
          }
        }
      }

      if (target_job_index < 0) {
        // no one wants to be scheduled..
        return success;
      }

      // skip this job if we can't schedule it immediately,
      // even if this job is the "most urgent" one
      const int worker_id = target_subgraph_key.GetWorkerId();
      if (idle_workers.find(worker_id) == idle_workers.end()) {
        auto maybe_expected_time = engine_.GetExpected(target_subgraph_key);
        if (!maybe_expected_time.has_value()) {
          BAND_LOG_PROD(BAND_LOG_ERROR,
                        "Failed to get expected latency for subgraph key %s",
                        target_subgraph_key.ToString().c_str());
          return false;
        }
        int64_t expected_time = maybe_expected_time.value();
        waiting_time[worker_id] += expected_time;
        auto requests_it = requests.begin() + target_job_index;
        Job job = *requests_it;
        jobs_to_yield.insert(job.id());
        continue;
      } else {
        break;
      }
    } while (true);

    auto requests_it = requests.begin() + target_job_index;
    Job job = *requests_it;

    // erase the job from requests and decrement window_size
    requests.erase(requests_it);
    window_size--;

    // Update Job status specific to this planner.
    // Common status will be updated by `EnqueueAction`.
    if (engine_.IsBegin(target_subgraph_key)) {
      // only set these fields if this is the first subgraph of this model
      job.expected_latency = largest_shortest_latency;
    }

    success &= engine_.EnqueueToWorker({job, target_subgraph_key});

    if (reserve_) {
      // add next job to reserved_, if one exists
      if (target_subgraph_key_next != SubgraphKey()) {
        reserved_[job.id()] = target_subgraph_key_next;
      } else {
        reserved_.erase(job.id());
      }
    }
  }
  return success;
}
}  // namespace band