#include "tensorflow/lite/planner/fixed_device_planner.h"

namespace tflite {
namespace impl {

void FixedDevicePlanner::Plan() {
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
      };
    }

    // The lock will not be released until the request queue is empty,
    // which means concurrent enqueue is not available.
    // This can affect the performance.
    std::unique_lock<std::mutex> lock(GetRequestsMtx());
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
      UpdateJobEnqueueStatus(to_execute, key);

      Worker* worker = GetInterpreter()->GetWorker(device_flag);
      if (worker->GiveJob(to_execute)) {
        UpdateJobWorkerStatus(to_execute, worker);
        // all is well
        // delete this job from our request queue
        it = requests.erase(it);
        sched_id_++;
      } else {
        // we couldn't assign this job to worker
        ++it;
      }
    }
    lock.unlock();
  }
}

bool FixedDevicePlanner::NeedProfile() {
  return false;
}

}  // namespace impl
}  // namespace tflite
