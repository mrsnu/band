#include "tensorflow/lite/planner/shortest_expected_latency_scheduler.h"

#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace impl {

void ShortestExpectedLatencyScheduler::Schedule(JobQueue& requests) {
  DeviceWaitingTime device_waiting = GetDeviceWaitingTime();
  JobQueue local_jobs;
  int window_size = std::min(planner_->GetWindowSize(), (int)requests.size());
  local_jobs.insert(local_jobs.begin(), requests.begin(),
                    requests.begin() + window_size);
  requests.erase(requests.begin(), requests.begin() + window_size);
  while (!local_jobs.empty()) {
    // First, find the most urgent job -- the one with the
    // largest shortest latency (no, that's not a typo).
    // Put that job into some worker, and repeat this whole loop until we've
    // gone through all jobs.
    // There should be a more quicker way do this, but I'm leaving this as-is
    // to make it simple.
    // E.g., we add interpreter.GetProfiledLatency() to the expected_latency map
    // of all Jobs instead of calling GetShortestLatency() a gazillion times
    // again.

    // Note that we are NOT considering enqueue_time at the moment;
    // no request is given higher priority even if it had stayed in the queue
    // for longer than others.

    // find the most urgent job and save its index within the queue
    int64_t largest_shortest_latency = -1;
    int target_job_idx;
    int target_subgraph_idx;

    int64_t sched_start = profiling::time::NowMicros();
    for (auto it = local_jobs.begin(); it != local_jobs.end(); ++it) {
      Job& next_job = *it;
      std::pair<int, int64_t> best_subgraph =
          GetInterpreter()->GetShortestLatency(
              next_job.model_id, next_job.resolved_tensors, 0, device_waiting);

      if (largest_shortest_latency < best_subgraph.second) {
        largest_shortest_latency = best_subgraph.second;
        target_job_idx = it - local_jobs.begin();
        target_subgraph_idx = best_subgraph.first;
      }
    }
    int64_t sched_end = profiling::time::NowMicros();
    // quick check for roughly examining the planning overhead
    // std::cout << "Time to Find the next job(us) : " <<  sched_end -
    // sched_start << std::endl;

    // for some reason, this Job must NOT be a reference (&), otherwise
    // we get a segfault at push_back() below
    Job most_urgent_job = local_jobs[target_job_idx];

    // remove the job from the queue so that we don't meet it in the next loop
    local_jobs.erase(local_jobs.begin() + target_job_idx);

    // Update Job status specific to this planner.
    // Common status will be updated by `EnqueueAction`.
    Subgraph* target_subgraph = GetInterpreter()->subgraph(target_subgraph_idx);
    if (target_subgraph->GetPrevSubgraph() == nullptr) {
      // only set these fields if this is the first subgraph of this model
      most_urgent_job.expected_latency = largest_shortest_latency;
    }
    EnqueueAction(most_urgent_job, target_subgraph);

    device_waiting[target_subgraph->GetKey().device_flag] +=
        largest_shortest_latency;
  }
}

}  // namespace impl
}  // namespace tflite
