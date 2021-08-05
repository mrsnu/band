#include "tensorflow/lite/planner/least_slack_first_scheduler.h"

#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace impl {

void LeastSlackFirstScheduler::Schedule(JobQueue& requests) {
  std::set<TfLiteDeviceFlags> idle_devices = planner_->GetIdleDevices();
  DeviceWaitingTime device_waiting = GetDeviceWaitingTime();

  SortBySlackTime(requests);
  for (auto it = requests.begin(); it != requests.end();) {
    if (idle_devices.empty()) {
      // no device is idle; wait for next iteration
      return;
    }
    Job next_job = *it;
    std::pair<int, int64_t> best_subgraph =
        GetInterpreter()->GetShortestLatency(
            next_job.model_id, next_job.resolved_tensors, 0, device_waiting);
    // if selected device is empty,
    Subgraph* target_subgraph = GetInterpreter()->subgraph(best_subgraph.first);

    // If the target device is not idle, give opportunity to the next job.
    TfLiteDeviceFlags device = target_subgraph->GetKey().device_flag;
    auto device_it = idle_devices.find(device);
    if (device_it == idle_devices.end()) {
      ++it;
      continue;
    }

    if (target_subgraph->GetPrevSubgraph() == nullptr) {
      // only set these fields if this is the first subgraph of this model
      next_job.expected_latency = best_subgraph.second;
    }
    EnqueueAction(next_job, target_subgraph);

    it = requests.erase(it);
    idle_devices.erase(device_it);
    device_waiting[device] +=
        GetInterpreter()->GetExpectedLatency(target_subgraph->GetKey());
  }
}

int64_t LeastSlackFirstScheduler::GetSlackTime(const Job& job) {
    return job.enqueue_time + job.slo_us - profiling::time::NowMicros();
}

void LeastSlackFirstScheduler::SortBySlackTime(JobQueue& requests) {
  std::sort(requests.begin(), requests.end(), [&] (const Job& first, const Job& second) -> bool {
    return GetSlackTime(first) < GetSlackTime(second);
  });
}

}  // namespace impl
}  // namespace tflite
