#include "tensorflow/lite/planner/fixed_device_planner.h"
#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace impl {

void FixedDeviceGlobalQueuePlanner::Plan() {
  while (true) {
    if (GetSafeBool().wait())
      return;

    std::set<int> models = GetInterpreter()->models();
    if (models.size() != model_device_map_.size()) {
      // (# of available devices, vector of model_id)
      std::map<int, std::set<int>> devices_per_models_map;
      for (auto model_id : models) {
        int count = 0;
        for (int device_idx = 0; device_idx < kTfLiteNumDevices; device_idx++) {
          if (GetInterpreter()->GetSubgraphIdx(
                model_id, static_cast<TfLiteDeviceFlags>(device_idx)) != -1) {
            count++;
          }
        }
        devices_per_models_map[count].insert(model_id);
      }

      int device_idx = 0;
      while (devices_per_models_map.size()) {
        // Loop through models in ascending order
        // based on # of available devices
        // (Assign models that has limited support first)
        int selected_model_id = -1;
        for (auto& devices_per_models : devices_per_models_map) {
          for (int model_id : devices_per_models.second) {
            if (GetInterpreter()->GetSubgraphIdx(
                  model_id, static_cast<TfLiteDeviceFlags>(device_idx)) != -1) {
              selected_model_id = model_id;
              break;
            }
          }

          if (selected_model_id != -1) {
            devices_per_models.second.erase(selected_model_id);
            if (devices_per_models.second.size() == 0)
              devices_per_models_map.erase(devices_per_models.first);
            break;
          }
        }

        if (selected_model_id != -1) {
          model_device_map_[selected_model_id] =
              static_cast<TfLiteDeviceFlags>(device_idx);
        }

        device_idx = (device_idx + 1) % kTfLiteNumDevices;
      }
    }

    std::set<TfLiteDeviceFlags> idle_devices;
    // for early-dropping requests that will miss their SLO
    std::map<TfLiteDeviceFlags, int64_t> device_waiting;
    for (int i = 0; i < kTfLiteNumDevices; ++i) {
      TfLiteDeviceFlags device_flag = static_cast<TfLiteDeviceFlags>(i);
      Worker* worker = GetInterpreter()->GetWorker(device_flag);
      if (worker != nullptr) {
        device_waiting[device_flag] = worker->GetWaitingTime();

        // we could, technically, check waiting time and isBusy with a single
        // function call if we slightly change Worker implementation
        if (!worker->IsBusy()) {
          idle_devices.insert(device_flag);
        }
      }
    }

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

      int64_t expected_execution_time = device_waiting[device_flag] + GetInterpreter()->GetExpectedLatency(key);

      UpdateJobEnqueueStatus(to_execute, key);
      to_execute.expected_execution_time = expected_execution_time;

      // this job has an SLO; check if it's not too late already
      if (to_execute.slo_us > 0) {
        int64_t current_time = profiling::time::NowMicros();

        if (current_time + expected_execution_time >
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
