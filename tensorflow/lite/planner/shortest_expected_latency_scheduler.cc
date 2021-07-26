#include "tensorflow/lite/planner/shortest_expected_latency_scheduler.h"
#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace impl {

ScheduleAction ShortestExpectedLatencyScheduler::Schedule(JobQueue& requests) {
  ScheduleAction action;
  DeviceWaitingTime device_waiting = planner_->device_waiting_;
  while (!requests.empty()) {
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
    int target_subgraph;

    int64_t sched_start = profiling::time::NowMicros();
    for (auto it = requests.begin(); it != requests.end(); ++it) {
      Job& next_job = *it;
      std::pair<int, int64_t> best_subgraph =
          planner_->GetInterpreter()->GetShortestLatency(
              next_job.model_id, next_job.start_idx, 0, device_waiting);

      if (largest_shortest_latency < best_subgraph.second) {
        largest_shortest_latency = best_subgraph.second;
        target_job_idx = it - requests.begin();
        target_subgraph = best_subgraph.first;
      }
    }
    int64_t sched_end = profiling::time::NowMicros();
    // quick check for roughly examining the planning overhead
    // std::cout << "Time to Find the next job(us) : " <<  sched_end -
    // sched_start << std::endl;

    // for some reason, this Job must NOT be a reference (&), otherwise
    // we get a segfault at push_back() below
    Job most_urgent_job = requests[target_job_idx];

    // remove the job from the queue so that we don't meet it in the next loop
    requests.erase(requests.begin() + target_job_idx);

    SubgraphKey& to_execute =
        planner_->GetInterpreter()->subgraph(target_subgraph)->GetKey();
    most_urgent_job.start_idx = to_execute.start_idx;
    most_urgent_job.end_idx = to_execute.end_idx;
    most_urgent_job.subgraph_idx = target_subgraph;
    most_urgent_job.device_id = to_execute.device_flag;
    most_urgent_job.profiled_time =
        planner_->GetInterpreter()->GetSubgraphProfileResult(to_execute);

    if (most_urgent_job.expected_latency == 0) {
      // only set these fields if this is the first subgraph of this model
      most_urgent_job.expected_latency = largest_shortest_latency;
      most_urgent_job.sched_id = planner_->sched_id_++;
    }

    ModelSpec& model_spec =
        planner_->GetInterpreter()->GetModelSpec(most_urgent_job.model_id);
    if (most_urgent_job.end_idx < model_spec.num_ops - 1) {
      Job remaining_ops(most_urgent_job.model_id);
      remaining_ops.enqueue_time = most_urgent_job.enqueue_time;
      remaining_ops.start_idx = most_urgent_job.end_idx + 1;
      remaining_ops.end_idx = model_spec.num_ops - 1;
      remaining_ops.following_jobs = most_urgent_job.following_jobs;
      remaining_ops.expected_latency = most_urgent_job.expected_latency;
      remaining_ops.sched_id = most_urgent_job.sched_id;
      remaining_ops.job_id = most_urgent_job.job_id;
      remaining_ops.input_handle = most_urgent_job.input_handle;
      remaining_ops.output_handle = most_urgent_job.output_handle;
      remaining_ops.previous_subgraph_idx = most_urgent_job.subgraph_idx;

      most_urgent_job.following_jobs.clear();
      most_urgent_job.following_jobs.push_back(remaining_ops);
      most_urgent_job.is_final_subgraph = false;
    }
    action[to_execute.device_flag].push_back(most_urgent_job);
    device_waiting[to_execute.device_flag] += largest_shortest_latency;
  }
  return action;
}

}  // namespace impl
}  // namespace tflite
