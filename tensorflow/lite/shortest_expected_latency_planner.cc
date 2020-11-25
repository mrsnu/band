#include "tensorflow/lite/shortest_expected_latency_planner.h"
#include "tensorflow/lite/profiling/time.h"
#include <iostream>

namespace tflite {
namespace impl {

void ShortestExpectedLatencyPlanner::Plan() {
  while (true) {
    if (GetSafeBool().wait())
      return;

    std::deque<Job> local_jobs;
    std::unique_lock<std::mutex> request_lock(GetRequestsMtx());
    if (!GetRequests().empty()) {
      // copy all elements to a local container so that
      // we can release the lock asap
      GetRequests().swap(local_jobs);
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
      // E.g., we add subgraph.GetExpectedLatency() to the expected_latency map
      // of all Jobs instead of calling GetShortestLatency() a gazillion times
      // again.

      // Note that we are NOT considering enqueue_time at the moment;
      // no request is given higher priority even if it had stayed in the queue
      // for longer than others.

      // find the most urgent job and save its index within the queue
      int64_t largest_shortest_latency = -1;
      int target_idx;
      TfLiteDevice target_device;
      for (auto it = local_jobs.begin(); it != local_jobs.end(); ++it) {
        Job& to_execute = *it;
        TfLiteDevice device = GetInterpreter()->GetShortestLatency(to_execute.model_id_, to_execute);
        int64_t shortest_latency = to_execute.expected_latency[device];

        if (shortest_latency > largest_shortest_latency) {
          largest_shortest_latency = shortest_latency;
          target_idx = it - local_jobs.begin();
          target_device = device;
        }
      }

      // for some reason, this Job must NOT be a reference (&), otherwise
      // we get a segfault at push_back() below
      Job most_urgent_job = local_jobs[target_idx];

      // remove the job from the queue so that we don't meet it in the next loop
      local_jobs.erase(local_jobs.begin() + target_idx);
      Worker& worker = GetInterpreter()->GetWorker(target_device);

      {
        std::lock_guard<std::mutex> lock(worker.GetDeviceMtx());
        int subgraph_idx = GetInterpreter()->GetSubgraphIdx(most_urgent_job.model_id_, target_device);
        most_urgent_job.subgraph_idx_ = subgraph_idx;
        most_urgent_job.device_id_ = target_device;

        worker.GetDeviceRequests().push_back(most_urgent_job);
        worker.GetRequestCv().notify_one();
      }
    }
  }
}

}  // namespace impl
}  // namespace tflite
