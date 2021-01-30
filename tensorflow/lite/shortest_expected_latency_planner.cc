#include "tensorflow/lite/shortest_expected_latency_planner.h"
#include <iostream>

namespace tflite {
namespace impl {

void ShortestExpectedLatencyPlanner::Plan() {
  int sched_id = 0;
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

    std::cout << "Got requests" << std::endl;
    while (!local_jobs.empty()) {
      // std::cout << local_jobs.size() << std::endl;
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
      for (auto it = local_jobs.begin(); it != local_jobs.end(); ++it) {
        Job& next_job = *it;
        std::vector<int> subgraph_indices =
          GetInterpreter()->GetSubgraphCandidates(next_job.model_id_,
                                                  next_job.start_idx);

        if (subgraph_indices.size() == 0) {
          // TODO(dhkim): error handling
          std::cout << "NO candidates are selected." << std::endl;
          return;
        }

        int64_t best_latency = INT_MAX;
        int best_subgraph = -1;
        for (auto subgraph_idx : subgraph_indices) {
          SubgraphKey& key = GetInterpreter()->subgraph(subgraph_idx)->GetKey();
          int64_t expected_latency =
            GetInterpreter()->GetShortestLatency(key, 0);
          if (expected_latency < best_latency) {
            best_latency = expected_latency;
            best_subgraph = subgraph_idx;
          }
        }

        /*
        std::cout << "For Job " << it - local_jobs.begin() << "(" << next_job.model_id_
                  << ", " << next_job.start_idx << ") : Best subgraph - "
                  << best_subgraph << ", " << best_latency << std::endl;
        */
        if (largest_shortest_latency < best_latency) {
          largest_shortest_latency = best_latency;
          target_job_idx = it - local_jobs.begin();
          target_subgraph = best_subgraph;
        }
      }

      // for some reason, this Job must NOT be a reference (&), otherwise
      // we get a segfault at push_back() below
      Job most_urgent_job = local_jobs[target_job_idx];

      // remove the job from the queue so that we don't meet it in the next loop
      local_jobs.erase(local_jobs.begin() + target_job_idx);

      SubgraphKey& to_execute =
          GetInterpreter()->subgraph(target_subgraph)->GetKey();
      most_urgent_job.start_idx = to_execute.start_idx;
      most_urgent_job.end_idx = to_execute.end_idx;
      most_urgent_job.subgraph_idx_ = target_subgraph;
      most_urgent_job.device_id_ = to_execute.device_flag;

      Worker* worker = GetInterpreter()->GetWorker(to_execute.device_flag);
      {
        std::lock_guard<std::mutex> lock(worker->GetDeviceMtx());
        most_urgent_job.sched_id_ = sched_id++;
        worker->GetDeviceRequests().push_back(most_urgent_job);
        worker->GetRequestCv().notify_one();
      }
    }
  }
}

bool ShortestExpectedLatencyPlanner::NeedProfile() {
  return true;
}
 
} // namespace impl
} // namespace tflite
