#include "tensorflow/lite/planner/shortest_expected_latency_planner.h"
#include "tensorflow/lite/profiling/time.h"

// for the std::cout commented out in Plan()
// #include <iostream>

namespace tflite {
namespace impl {

void ShortestExpectedLatencyPlanner::Plan() {
  while (true) {
    if (GetSafeBool().wait()) return;

    JobQueue local_jobs = CopyToLocalQueue();

    while (!local_jobs.empty()) {
      // First, find the most urgent job -- the one with the
      // largest shortest latency (no, that's not a typo).
      // Put that job into some worker, and repeat this whole loop until we've
      // gone through all jobs.
      // There should be a more quicker way do this, but I'm leaving this as-is
      // to make it simple.
      // E.g., we add interpreter.GetExpectedLatency() to the expected_latency map
      // of all Jobs instead of calling GetShortestLatency() a gazillion times
      // again.

      // Note that we are NOT considering enqueue_time at the moment;
      // no request is given higher priority even if it had stayed in the queue
      // for longer than others.

      // first, get per-device waiting times
      std::map<TfLiteDeviceFlags, int64_t> device_waiting_time;
      for (int i = 0; i < kTfLiteNumDevices; ++i) {
        TfLiteDeviceFlags device_flag = static_cast<TfLiteDeviceFlags>(i);
        Worker* worker = GetInterpreter()->GetWorker(device_flag);
        if (worker != nullptr) {
          device_waiting_time[device_flag] = worker->GetWaitingTime();
        }
      }

      // find the most urgent job and save its index within the queue
      int64_t largest_shortest_latency = -1;
      int target_job_idx;
      int target_subgraph_idx;

      int64_t sched_start = profiling::time::NowMicros();
      for (auto it = local_jobs.begin(); it != local_jobs.end(); ++it) {
        Job& next_job = *it;
        std::set<int> resolved_tensors = next_job.resolved_tensors;

        std::pair<int, int64_t> best_subgraph =
            interpreter_->GetShortestLatency(next_job.model_id, resolved_tensors, 0, device_waiting_time);

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

      Subgraph* target_subgraph =
          GetInterpreter()->subgraph(target_subgraph_idx);
      SubgraphKey& to_execute = target_subgraph->GetKey();
      UpdateJobEnqueueStatus(most_urgent_job, to_execute);

      if (most_urgent_job.expected_latency == 0) {
        // only set these fields if this is the first subgraph of this model
        most_urgent_job.expected_latency = largest_shortest_latency;
        most_urgent_job.sched_id = sched_id_++;
      }

      // this job has an SLO; check if it's not too late already
      if (most_urgent_job.slo_us > 0) {
        int64_t current_time = profiling::time::NowMicros();
        int64_t expected_latency =
            device_waiting_time[to_execute.device_flag] +
            most_urgent_job.expected_execution_time;

        if (current_time + expected_latency >
            most_urgent_job.enqueue_time + most_urgent_job.slo_us) {
          // SLO violation
          // no point in running this job anymore
          most_urgent_job.status = kTfLiteJobSLOViolation;

          // mark this as -1 to differentiate it from the default value, 0
          most_urgent_job.invoke_time = -1;

          // mark the time of this decision (of early-dropping this job)
          most_urgent_job.end_time = current_time;
          EnqueueFinishedJob(most_urgent_job);
          continue;
        }
      }

      ModelSpec& model_spec =
          GetInterpreter()->GetModelSpec(most_urgent_job.model_id);
      if (target_subgraph->GetNextSubgraph() != nullptr) {
        Job remaining_ops(most_urgent_job.model_id);
        remaining_ops.enqueue_time = most_urgent_job.enqueue_time;
        remaining_ops.following_jobs = most_urgent_job.following_jobs;
        remaining_ops.expected_latency = most_urgent_job.expected_latency;
        remaining_ops.sched_id = most_urgent_job.sched_id;
        remaining_ops.job_id = most_urgent_job.job_id;
        remaining_ops.input_handle = most_urgent_job.input_handle;
        remaining_ops.output_handle = most_urgent_job.output_handle;
        remaining_ops.previous_subgraph_idx = most_urgent_job.subgraph_idx;
        remaining_ops.resolved_tensors = most_urgent_job.resolved_tensors;

        for (int output_index : target_subgraph->outputs()) {
          remaining_ops.resolved_tensors.insert(output_index);
        }

        most_urgent_job.following_jobs.clear();
        most_urgent_job.following_jobs.push_back(remaining_ops);
      }

      Worker* worker = GetInterpreter()->GetWorker(to_execute.device_flag);
      {
        std::lock_guard<std::mutex> lock(worker->GetDeviceMtx());
        worker->GetDeviceRequests().push_back(most_urgent_job);
        worker->GetRequestCv().notify_one();
      }
    }
  }
}

bool ShortestExpectedLatencyPlanner::NeedProfile() { return true; }

}  // namespace impl
}  // namespace tflite
