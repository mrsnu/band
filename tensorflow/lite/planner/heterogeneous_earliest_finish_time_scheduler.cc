#include "tensorflow/lite/planner/heterogeneous_earliest_finish_time_scheduler.h"

namespace tflite {
namespace impl {

void HeterogeneousEarliestFinishTimeScheduler::Schedule(JobQueue& requests) {
  int window_size = std::min(planner_->GetWindowSize(), (int)requests.size());
  // stop if there are no idle devices OR there's nothing in `requests`
  while (window_size > 0) {
    planner_->UpdateWorkerWaitingTime();
    std::set<int> idle_workers = planner_->GetIdleWorkers();
    if (idle_workers.empty()) {
      break;
    }
    // Lookup table for GetShortestLatency().
    // Although it is tempting to maintain a scheduler-wide cache,
    // the values in device_waiting are (generally) different for every while
    // loop iteration so this cache is valid only for this specific loop iter.
    //
    // Although this cache looks the same as the one inside
    // GetShortestLatency(), we can get slightly more cache hits by having
    // it again here. The reason is because we know that `device_waiting`
    // values are unchanged within this loop iter, so if two Jobs have the
    // same model_id and resolved_tensors then GetShortestLatency() will
    // return the same value. In contrast, GetShortestLatency() always skips
    // the cache lookup if start_time is not larger than all `device_waiting`
    // values, even if the same `device_waiting` values are given.
    std::unordered_map<std::pair<int, std::set<int>>,
                       std::pair<int, int64_t>,
                       Interpreter::PairHash> cache;

    // basically the same as ShortestExpectedLatencyScheduler
    int64_t largest_shortest_latency = -1;
    int target_job_idx = -1;
    int target_subgraph_idx = -1;

    // only check up to `window_size` requests
    for (auto it = requests.begin(); it != requests.begin() + window_size; ++it) {
      Job& job = *it;
      int preceded_subgraph_index = job.previous_subgraph_indices.empty() ?
                                    -1 : job.previous_subgraph_indices.back();

      std::pair<int, int64_t> best_subgraph = {-1, INT_MAX};
      std::pair<int, std::set<int>> cache_key = {job.model_id,
                                                 job.resolved_tensors};

      auto cache_it = cache.find(cache_key);
      if (cache_it != cache.end()) {
        // used cached value instead of calling GetShortestLatency()
        best_subgraph = cache_it->second;
      } else {
        best_subgraph = GetInterpreter()->GetShortestLatency(
            job.model_id, job.resolved_tensors, 0, GetWorkerWaitingTime(),
            preceded_subgraph_index);

        // insert new value into cache
        cache[cache_key] = best_subgraph;
      }


      if (largest_shortest_latency < best_subgraph.second) {
        Subgraph* target_subgraph = GetInterpreter()->subgraph(best_subgraph.first);

        // skip this job if we can't schedule it immediately,
        // even if this job is the "most urgent" one
        if (idle_workers.find(target_subgraph->GetKey().worker_id) ==
            idle_workers.end()) {
          continue;
        }

        largest_shortest_latency = best_subgraph.second;
        target_job_idx = it - requests.begin();
        target_subgraph_idx = best_subgraph.first;
      }
    }

    if (target_job_idx < 0) {
      // no one wants to be scheduled..
      break;
    }

    auto requests_it = requests.begin() + target_job_idx;
    Job job = *requests_it;

    // erase the job from requests and decrement window_size
    requests.erase(requests_it);
    window_size--;

    Subgraph* target_subgraph = GetInterpreter()->subgraph(target_subgraph_idx);
    auto worker_it = idle_workers.find(target_subgraph->GetKey().worker_id);
    assert(worker_it != idle_workers.end());

    // Update Job status specific to this planner.
    // Common status will be updated by `EnqueueAction`.
    if (target_subgraph->IsStart()) {
      // only set these fields if this is the first subgraph of this model
      job.expected_latency = largest_shortest_latency;
    }
    EnqueueAction(job, target_subgraph);
  }
}

}  // namespace impl
}  // namespace tflite
