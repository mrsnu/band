#include "tensorflow/lite/planner/heterogeneous_earliest_finish_time_scheduler.h"

#include "tensorflow/lite/profiling/time.h"

#include <iostream>

namespace tflite {
namespace impl {

void HeterogeneousEarliestFinishTimeScheduler::Schedule(JobQueue& requests) {
  std::set<TfLiteDeviceFlags> idle_devices = planner_->GetIdleDevices();
  DeviceWaitingTime device_waiting = GetDeviceWaitingTime();
  int window_size = std::min(planner_->GetWindowSize(), (int)requests.size());

  while (!idle_devices.empty() && window_size > 0) {
    std::unordered_map<std::tuple<int, std::set<int>, int>, std::pair<int, int64_t>, TupleHash> cache;

    int64_t largest_shortest_latency = -1;
    int target_job_idx = -1;
    int target_subgraph_idx = -1;

    for (auto it = requests.begin(); it != requests.begin() + window_size; ++it) {
      Job& job = *it;
      int preceded_subgraph_index = job.previous_subgraph_indices.empty() ?
                                    -1 : job.previous_subgraph_indices.back();

      std::pair<int, int64_t> best_subgraph = {-1, INT_MAX};
      std::tuple<int, std::set<int>, int> cache_key =
          {job.model_id, job.resolved_tensors, preceded_subgraph_index};

      auto cache_it = cache.find(cache_key);
      if (cache_it != cache.end()) {
        best_subgraph = cache_it->second;
      } else {
        best_subgraph = GetInterpreter()->GetShortestLatency(
            job.model_id, job.resolved_tensors, 0, device_waiting,
            preceded_subgraph_index);

        cache[cache_key] = best_subgraph;
      }


      if (largest_shortest_latency < best_subgraph.second) {
        Subgraph* target_subgraph = GetInterpreter()->subgraph(best_subgraph.first);
        TfLiteDeviceFlags device = target_subgraph->GetKey().device_flag;
        if (idle_devices.find(device) == idle_devices.end()) {
          continue;
        }

        largest_shortest_latency = best_subgraph.second;
        target_job_idx = it - requests.begin();
        target_subgraph_idx = best_subgraph.first;
      }
    }


    if (target_job_idx < 0) {
      break;
    }


    auto requests_it = requests.begin() + target_job_idx;
    Job job = *requests_it;
    requests.erase(requests_it);
    window_size--;

    Subgraph* target_subgraph = GetInterpreter()->subgraph(target_subgraph_idx);
    TfLiteDeviceFlags device = target_subgraph->GetKey().device_flag;
    auto device_it = idle_devices.find(device);
    assert(device_it != idle_devices.end());
    idle_devices.erase(device_it);

    device_waiting[device] +=
        GetInterpreter()->GetExpectedLatency(target_subgraph->GetKey());
    if (target_subgraph->IsStart()) {
      // only set these fields if this is the first subgraph of this model
      job.expected_latency = largest_shortest_latency;
    }
    EnqueueAction(job, target_subgraph);
  }
}

}  // namespace impl
}  // namespace tflite
