#include "band/scheduler/heterogeneous_earliest_finish_time_scheduler.h"

#include <unordered_set>

#include "band/logger.h"

namespace band {
HEFTScheduler::HEFTScheduler(IEngine& engine, int window_size, bool reserve)
    : IScheduler(engine), window_size_(window_size), reserve_(reserve) {}

bool HEFTScheduler::Schedule(JobQueue& requests) {
  BAND_LOG_PROD(BAND_LOG_INFO, "HEFTScheduler::Schedule");
  bool success = true;
  int num_jobs = std::min(window_size_, (int)requests.size());
  while (num_jobs > 0) {
    BAND_LOG_PROD(BAND_LOG_INFO, "HEFTScheduler::Schedule: num_jobs = %d",
                  num_jobs);
    engine_.UpdateWorkersWaiting();

    // stop if there are no idle devices.
    std::set<int> idle_workers = engine_.GetIdleWorkers();
    if (idle_workers.empty()) {
      break;
    }

    // hold on to a local copy of worker waiting time
    WorkerWaitingTime waiting_time = engine_.GetWorkerWaitingTime();
    std::set<JobId> jobs_to_yield;

    // basically the same as ShortestExpectedLatencyScheduler
    double largest_shortest_latency;
    int target_job_index;
    SubgraphKey target_subgraph_key;
    SubgraphKey target_subgraph_key_next;

    do {
      largest_shortest_latency = -1;
      target_job_index = -1;

      // only check up to `num_jobs` requests
      std::unordered_set<std::pair<int, BitMask>, JobIdBitMaskHash>
          searched_jobs;
      for (auto it = requests.begin(); it != requests.begin() + num_jobs;
           ++it) {
        Job job = *it;

        if (jobs_to_yield.find(job.job_id) != jobs_to_yield.end()) {
          continue;
        }

        std::pair<ModelId, BitMask> job_to_search =
            std::make_pair(job.model_id, job.resolved_unit_subgraphs);
        if (searched_jobs.find(job_to_search) != searched_jobs.end()) {
          continue;
        }
        
        searched_jobs.insert(job_to_search);

        // update waiting_time for all future jobs in reserved_
        WorkerWaitingTime reserved_time(waiting_time);
        for (auto job_subgraph_key : reserved_) {
          if (job_subgraph_key.first == job.job_id) {
            continue;
          }

          reserved_time[job_subgraph_key.second.GetWorkerId()] +=
              engine_.GetExpected(job_subgraph_key.second);
        }

        std::pair<std::vector<SubgraphKey>, double> best_subgraph =
            engine_.GetSubgraphWithShortestLatency(job, reserved_time);
        BAND_LOG_PROD(BAND_LOG_INFO,
                      "HEFTScheduler::Schedule: best_subgraph.first = %s",
                      best_subgraph.first[0].ToString().c_str());
        BAND_LOG_PROD(BAND_LOG_INFO,
                      "HEFTScheduler::Schedule: best_subgraph.second = %f",
                      best_subgraph.second);

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

      BAND_LOG_PROD(BAND_LOG_INFO,
                    "HEFTScheduler::Schedule: largest_shortest_latency = %f",
                    largest_shortest_latency);
      BAND_LOG_PROD(BAND_LOG_INFO, "HEFTScheduler::Schedule: target_job_index = %d",
                    target_job_index);

      // no one wants to be scheduled.
      if (target_job_index < 0) {
        return success;
      }

      // skip this job if we can't schedule it immediately,
      // even if this job is the "most urgent" one
      const int worker_id = target_subgraph_key.GetWorkerId();
      if (idle_workers.find(worker_id) != idle_workers.end()) {
        BAND_LOG_PROD(BAND_LOG_INFO,
                      "HEFTScheduler::Schedule: idle_workers.find(worker_id) "
                      "!= idle_workers.end()");
        break;
      }
      waiting_time[worker_id] += engine_.GetExpected(target_subgraph_key);
      BAND_LOG_PROD(BAND_LOG_INFO,
                    "HEFTScheduler::Schedule: waiting_time[worker_id] = %f",
                    waiting_time[worker_id]);
      auto requests_it = requests.begin() + target_job_index;
      Job job = *requests_it;
      jobs_to_yield.insert(job.job_id);
    } while (true);

    auto requests_it = requests.begin() + target_job_index;
    Job job = *requests_it;

    // erase the job from requests and decrement num_jobs
    requests.erase(requests_it);
    num_jobs--;

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
        reserved_[job.job_id] = target_subgraph_key_next;
      } else {
        reserved_.erase(job.job_id);
      }
    }
  }

  if (!success) {
    BAND_LOG_PROD(BAND_LOG_ERROR, "HEFTScheduler failed to schedule");
  }
  return success;
}
}  // namespace band