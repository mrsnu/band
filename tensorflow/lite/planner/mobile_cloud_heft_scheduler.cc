#include "tensorflow/lite/planner/mobile_cloud_heft_scheduler.h"
#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "libtflite", __VA_ARGS__)
#include <android/log.h>
#endif // defined(__ANDROID__)
namespace tflite {
namespace impl {

void MobileCloudHeftScheduler::Schedule(JobQueue& requests) {
  int window_size = std::min(planner_->GetWindowSize(), (int)requests.size());
  // stop if there are no idle devices OR there's nothing in `requests`
  while (window_size > 0) {
    planner_->UpdateWorkerWaitingTime();
    std::set<int> idle_workers = planner_->GetIdleAllWorkers();
    if (idle_workers.empty()) {
      break;
    }

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

      // only check up to `window_size` requests
      std::set<std::pair<int, int>> searched_jobs;
      for (auto it = requests.begin(); it != requests.begin() + window_size; ++it) {
        Job& job = *it;

        if (jobs_to_yield.find(job.job_id) != jobs_to_yield.end()) {
          continue;
        }

        std::pair<int, int> job_to_search = std::make_pair(job.model_id, job.start_unit_idx);
        if (searched_jobs.find(job_to_search) != searched_jobs.end()) {
          continue;
        } else {
          searched_jobs.insert(job_to_search);
        }

        // update waiting_time for all future jobs in reserved_
        WorkerWaitingTime reserved_time(waiting_time);
        for (auto pair : reserved_) {
          if (pair.first == job.job_id) {
            continue;
          }

          int reserved_id = pair.second;
          Subgraph* reserved_subgraph = GetInterpreter()->subgraph(reserved_id);
          int worker_id = reserved_subgraph->GetKey().worker_id;
          int64_t latency = model_manager_->GetPredictedLatency(worker_id, reserved_subgraph);
          reserved_time[worker_id] += latency;
        }

        std::pair<int, int64_t> best_subgraph = GetShortestSubgraph(job.model_id, waiting_time);

        if (largest_shortest_latency < best_subgraph.second) {
          Subgraph* target_subgraph = GetInterpreter()->subgraph(best_subgraph.first);

          largest_shortest_latency = best_subgraph.second;
          target_subgraph_idx = best_subgraph.first;
          target_job_idx = it - requests.begin();
        }
      }

      if (target_job_idx < 0) {
        // no one wants to be scheduled..
        return;
      }

      // skip this job if we can't schedule it immediately,
      // even if this job is the "most urgent" one
      Subgraph* target_subgraph = GetInterpreter()->subgraph(target_subgraph_idx);
      int worker_id = target_subgraph->GetKey().worker_id;
      if (idle_workers.find(worker_id) == idle_workers.end()) {
        waiting_time[worker_id] += model_manager_->GetPredictedLatency(worker_id, target_subgraph);
        auto requests_it = requests.begin() + target_job_idx;
        Job job = *requests_it;
        jobs_to_yield.insert(job.job_id);
        continue;
      } else {
        break;
      }
    } while (true);

    auto requests_it = requests.begin() + target_job_idx;
    Job job = *requests_it;

    // erase the job from requests and decrement window_size
    requests.erase(requests_it);
    window_size--;

    Subgraph* target_subgraph = GetInterpreter()->subgraph(target_subgraph_idx);
    job.estimated_latency = largest_shortest_latency;
    job.estimated_temp = 0;
    EnqueueAction(job, target_subgraph);

    reserved_.erase(job.job_id);
  }
}

std::pair<int, int64_t>
MobileCloudHeftScheduler::GetShortestSubgraph(int model_id, std::map<int, int64_t>& worker_waiting) {
  int64_t min_latency = INT64_MAX;
  int min_idx = -1;

  std::vector<int> subgraph_indices = GetInterpreter()->GetSubgraphIndices(model_id);
  for (auto subgraph_index : subgraph_indices) {
    Subgraph* subgraph = GetInterpreter()->subgraph(subgraph_index);
    SubgraphKey& key = subgraph->GetKey();

    int64_t waiting_time = worker_waiting[key.worker_id];
    int64_t expected_latency = model_manager_->GetPredictedLatency(key.worker_id, subgraph);
    int64_t total = expected_latency + waiting_time;

    if (min_latency > total) {
      min_latency = total;
      min_idx = subgraph_index;
    }
  }
  return {min_idx, min_latency};
}


}  // namespace impl
}  // namespace tflite