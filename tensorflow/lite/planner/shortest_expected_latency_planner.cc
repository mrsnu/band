#include "tensorflow/lite/planner/shortest_expected_latency_planner.h"
#include "tensorflow/lite/profiling/time.h"

// for the std::cout commented out in Plan()
// #include <iostream>

namespace tflite {
namespace impl {

void ShortestExpectedLatencyPlanner::Plan() {
  int sched_id = 0;
  while (true) {
    if (GetSafeBool().wait())
      return;

    std::deque<Job> local_jobs;
    std::unique_lock<std::mutex> request_lock(GetRequestsMtx());
    if (!requests_.empty()) {
      // Gets the specific amount of jobs from requests
      // and removes those jobs from the requests_.
      int window_size = std::min(GetWindowSize(), (int) requests_.size());
      local_jobs.insert(local_jobs.begin(), requests_.begin(), requests_.begin() + window_size);
      requests_.erase(requests_.begin(), requests_.begin() + window_size);
    } else {
      continue;
    }
    request_lock.unlock();

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
      int target_subgraph;

      int64_t sched_start = profiling::time::NowMicros();
      for (auto it = local_jobs.begin(); it != local_jobs.end(); ++it) {
        Job& next_job = *it;
        std::pair<int, int64_t> best_subgraph =
            GetInterpreter()->GetShortestLatency(next_job.model_id,
                                                 next_job.start_idx,
                                                 0,
                                                 device_waiting_time);

        if (largest_shortest_latency < best_subgraph.second) {
          largest_shortest_latency = best_subgraph.second;
          target_job_idx = it - local_jobs.begin();
          target_subgraph = best_subgraph.first;
        }
      }
      int64_t sched_end = profiling::time::NowMicros();
      // quick check for roughly examining the planning overhead
      // std::cout << "Time to Find the next job(us) : " <<  sched_end - sched_start << std::endl;

      // for some reason, this Job must NOT be a reference (&), otherwise
      // we get a segfault at push_back() below
      Job most_urgent_job = local_jobs[target_job_idx];

      // remove the job from the queue so that we don't meet it in the next loop
      local_jobs.erase(local_jobs.begin() + target_job_idx);

      SubgraphKey& to_execute =
          GetInterpreter()->subgraph(target_subgraph)->GetKey();
      most_urgent_job.start_idx = to_execute.start_idx;
      most_urgent_job.end_idx = to_execute.end_idx;
      most_urgent_job.subgraph_idx = target_subgraph;
      most_urgent_job.device_id = to_execute.device_flag;
      most_urgent_job.sched_id = sched_id++;

      ModelSpec& model_spec =
          GetInterpreter()->GetModelSpec(most_urgent_job.model_id);
      if (most_urgent_job.end_idx < model_spec.num_ops - 1) {
        Job remaining_ops(most_urgent_job.model_id);
        remaining_ops.enqueue_time = most_urgent_job.enqueue_time;
        remaining_ops.start_idx = most_urgent_job.end_idx + 1;
        remaining_ops.end_idx = model_spec.num_ops - 1;
        remaining_ops.following_jobs = most_urgent_job.following_jobs;

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

bool ShortestExpectedLatencyPlanner::NeedProfile() {
  return true;
}

}  // namespace impl
}  // namespace tflite
