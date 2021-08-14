#include "tensorflow/lite/planner/shortest_expected_latency_scheduler.h"

#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace impl {

void ShortestExpectedLatencyScheduler::Schedule(JobQueue& requests) {
  DeviceWaitingTime device_waiting = GetDeviceWaitingTime();
  JobQueue local_jobs;
  int window_size = std::min(planner_->GetWindowSize(), (int)requests.size());
  // local_jobs.insert(local_jobs.begin(), requests.begin(),
  //                   requests.begin() + window_size);
  // requests.erase(requests.begin(), requests.begin() + window_size);

  while (window_size > 0) {
    // find the most urgent job and save its index within the queue
    int64_t largest_shortest_latency = -1;
    int target_job_idx = -1;
    int target_subgraph_idx = -1;

    // Lookup table for GetShortestLatency().
    // See HeterogeneousEarliestFinishTimeScheduler for detailed comments.
    std::unordered_map<std::pair<int, std::set<int>>, std::pair<int, int64_t>, Interpreter::PairHash> cache;

    int64_t sched_start = profiling::time::NowMicros();
    for (auto it = requests.begin(); it != requests.begin() + window_size; it++) {
      Job& next_job = *it;
      if (active_job_ids_.size() >= 3 &&
          active_job_ids_.find(next_job.job_id) == active_job_ids_.end()) {
        continue;
      }

      int preceded_subgraph_index =
          next_job.previous_subgraph_indices.empty()
              ? -1
              : next_job.previous_subgraph_indices.back();
      std::pair<int, int64_t> best_subgraph;
      std::pair<int, std::set<int>> cache_key = {next_job.model_id,
                                                 next_job.resolved_tensors};

      auto cache_it = cache.find(cache_key);
      if (cache_it != cache.end()) {
        // used cached value instead of calling GetShortestLatency()
        best_subgraph = cache_it->second;
      } else {

        best_subgraph = GetInterpreter()->GetShortestLatency(
            next_job.model_id, next_job.resolved_tensors, 0, device_waiting,
            preceded_subgraph_index);

        // insert new value into cache
        cache[cache_key] = best_subgraph;
      }

      if (largest_shortest_latency < best_subgraph.second) {
        largest_shortest_latency = best_subgraph.second;
        target_job_idx = it - requests.begin();
        target_subgraph_idx = best_subgraph.first;
      }
    }
    int64_t sched_end = profiling::time::NowMicros();

    if (target_job_idx < 0) {
      break;
    }

    // quick check for roughly examining the planning overhead
    // std::cout << "Time to Find the next job(us) : " <<  sched_end -
    // sched_start << std::endl;

    // for some reason, this Job must NOT be a reference (&), otherwise
    // we get a segfault at push_back() below
    Job most_urgent_job = requests[target_job_idx];

    // remove the job from the queue so that we don't meet it in the next loop
    requests.erase(requests.begin() + target_job_idx);
    window_size--;

    // Update Job status specific to this planner.
    // Common status will be updated by `EnqueueAction`.
    Subgraph* target_subgraph = GetInterpreter()->subgraph(target_subgraph_idx);
    if (target_subgraph->IsStart()) {
      // only set these fields if this is the first subgraph of this model
      most_urgent_job.expected_latency = largest_shortest_latency;
      auto it = active_job_ids_.find(most_urgent_job.job_id);
      if (it != active_job_ids_.end()) {
        std::cout << "SEL: IsStart but job was already active" << std::endl;
        assert(false);
      }
      active_job_ids_.insert(most_urgent_job.job_id);

    }
    if (target_subgraph->IsEnd()) {
      auto it = active_job_ids_.find(most_urgent_job.job_id);
      if (it == active_job_ids_.end()) {
        std::cout << "SEL: IsEnd but job was not active" << std::endl;
        assert(false);
      }
      active_job_ids_.erase(it);
    }
    EnqueueAction(most_urgent_job, target_subgraph);

    device_waiting[target_subgraph->GetKey().device_flag] +=
        largest_shortest_latency;
  }
}

}  // namespace impl
}  // namespace tflite
