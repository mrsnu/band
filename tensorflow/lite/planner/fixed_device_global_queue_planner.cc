#include "tensorflow/lite/planner/fixed_device_planner.h"
#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace impl {

void FixedDeviceGlobalQueuePlanner::Plan() {
  while (true) {
    if (GetSafeBool().wait())
      return;

    UpdateModelDeviceMapping();
    UpdateDeviceWaitingTime();
    std::set<TfLiteDeviceFlags> idle_devices = GetIdleDevices();
    if (idle_devices.empty()) {
      // no device is idle; wait for next iteration
      // technically, we can skip this segment here because we check
      // idle_devices below anyway, but by exiting early we can avoid
      // acquiring the lock
      continue;
    }

    // The lock will not be released until the request queue is empty,
    // which means concurrent enqueue is not available.
    // This can affect the performance.
    std::lock_guard<std::mutex> lock(GetRequestsMtx());
    JobQueue& requests = GetRequests();
    for (auto it = requests.begin(); it != requests.end();) {
      Job& to_execute = *it;
      int model_id = to_execute.model_id;

      int device_idx;
      if (kTfLiteCPU <= to_execute.device_id &&
          to_execute.device_id < kTfLiteNumDevices) {
        device_idx = to_execute.device_id;
      } else {
        device_idx = model_device_map_[model_id];
      }
      TfLiteDeviceFlags device_flag =
          static_cast<TfLiteDeviceFlags>(device_idx);

      // TODO: fallback subgraphs for FixedDevicePlanner?
      int subgraph_idx = GetInterpreter()->GetSubgraphIdx(model_id, device_flag);
      SubgraphKey& key = GetInterpreter()->subgraph(subgraph_idx)->GetKey();

      int64_t profiled = GetInterpreter()->GetSubgraphProfileResult(key);
      int64_t expected_latency = device_waiting_[device_flag] + profiled;

      to_execute.profiled_time = profiled;
      to_execute.expected_latency = expected_latency;

      // this job has an SLO; check if it's not too late already
      if (to_execute.slo_us > 0) {
        int64_t current_time = profiling::time::NowMicros();

        if (current_time + expected_latency >
            to_execute.enqueue_time + to_execute.slo_us) {
          // SLO violation
          // there is no hope left for this job, throw it away
          to_execute.status = kTfLiteJobSLOViolation;

          // mark this as -1 to differentiate it from the default value, 0
          to_execute.invoke_time = -1;

          // mark the time of this decision (of early-dropping this job)
          to_execute.end_time = current_time;
          to_execute.sched_id = sched_id_++;
          EnqueueFinishedJob(to_execute);
          it = requests.erase(it);
          continue;
        }
      }

      auto idle_devices_it = idle_devices.find(device_flag);
      if (idle_devices_it == idle_devices.end()) {
        // that device is not idle, so leave this job alone for now
        ++it;
        continue;
      }

      to_execute.subgraph_idx = subgraph_idx;
      to_execute.device_id = device_idx;
      to_execute.sched_id = sched_id_++;

      Worker* worker = GetInterpreter()->GetWorker(device_flag);
      if (!worker->GiveJob(to_execute)) {
        // for some reason, the worker was busy and we couldn't assign
        // this job to it
        ++it;
        continue;
      }

      // all is well
      // delete this job from our request queue and
      // delete this device from our idle_devices set
      it = requests.erase(it);
      idle_devices.erase(idle_devices_it);

      if (idle_devices.empty()) {
        // no device is idle; wait for next iteration
        break;
      }
    }
  }
}

bool FixedDeviceGlobalQueuePlanner::NeedProfile() {
  // Required for checking SLO violation.
  // We could add an option to this planner for skipping the SLO check,
  // in which case this function can return false.
  return true;
}

}  // namespace impl
}  // namespace tflite
