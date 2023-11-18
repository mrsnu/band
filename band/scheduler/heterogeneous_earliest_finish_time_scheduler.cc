#include "band/scheduler/heterogeneous_earliest_finish_time_scheduler.h"

#include <unordered_set>

#include "band/logger.h"

namespace band {
HEFTScheduler::HEFTScheduler(IEngine& engine, int window_size, bool reserve)
    : IScheduler(engine), window_size_(window_size), reserve_(reserve) {}

bool HEFTScheduler::Schedule(JobQueue& requests) {
  bool success = true;
  int num_jobs = std::min(window_size_, (int)requests.size());
  while (num_jobs > 0) {
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
    double largest_min_cost = -1;
    double largest_expected_latency = -1;
    int target_job_index;
    SubgraphKey target_subgraph_key;
    SubgraphKey target_subgraph_key_next;

    do {
      largest_min_cost = -1;
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

        double expected_lat;
        std::pair<std::vector<SubgraphKey>, double> best_subgraph =
            engine_.GetSubgraphWithMinCost(
                job, reserved_time,
                [&expected_lat](double lat, std::map<SensorFlag, double>) -> double {
                  expected_lat = lat;
                  return lat;
                });

        if (largest_min_cost < best_subgraph.second) {
          largest_min_cost = best_subgraph.second;
          largest_expected_latency = expected_lat;
          target_subgraph_key = best_subgraph.first.front();
          target_job_index = it - requests.begin();
          if (best_subgraph.first.size() > 1) {
            target_subgraph_key_next = best_subgraph.first[1];
          } else {
            target_subgraph_key_next = {};
          }
        }
      }

      // no one wants to be scheduled.
      if (target_job_index < 0) {
        return success;
      }

      // skip this job if we can't schedule it immediately,
      // even if this job is the "most urgent" one
      const int worker_id = target_subgraph_key.GetWorkerId();
      if (idle_workers.find(worker_id) != idle_workers.end()) {
        break;
      }
      waiting_time[worker_id] += engine_.GetExpected(target_subgraph_key);
      auto requests_it = requests.begin() + target_job_index;
      Job& job = *requests_it;

      jobs_to_yield.insert(job.job_id);
    } while (true);

    auto requests_it = requests.begin() + target_job_index;
    Job job = *requests_it;

    // erase the job from requests and decrement num_jobs
    requests.erase(requests_it);
    num_jobs--;

    job.expected_latency = largest_expected_latency;

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

  return success;
}
}  // namespace band