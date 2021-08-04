#include "tensorflow/lite/planner/least_slack_first_scheduler.h"

#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace impl {

void LeastSlackFirstScheduler::Schedule(JobQueue& requests) {
  DeviceWaitingTime device_waiting = GetDeviceWaitingTime();
  JobQueue local_jobs;
  int window_size = std::min(planner_->GetWindowSize(), (int)requests.size());
  local_jobs.insert(local_jobs.begin(), requests.begin(),
                    requests.begin() + window_size);
  requests.erase(requests.begin(), requests.begin() + window_size);
  while (!local_jobs.empty()) {
    // ...
    Job most_urgent_job = local_jobs[target_job_idx];

    // remove the job from the queue so that we don't meet it in the next loop
    local_jobs.erase(local_jobs.begin() + target_job_idx);

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
