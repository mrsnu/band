#include "tensorflow/lite/planner/fixed_device_planner.h"

namespace tflite {
namespace impl {

void FixedDevicePlanner::Plan() {
  while (true) {
    if (GetSafeBool().wait())
      return;

    UpdateModelDeviceMapping();
    // The lock will not be released until the request queue is empty,
    // which means concurrent enqueue is not available.
    // This can affect the performance.
    std::unique_lock<std::mutex> lock(GetRequestsMtx());
    while (!GetRequests().empty()) {
      Job to_execute = GetRequests().front();
      GetRequests().pop_front();

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
      to_execute.subgraph_idx = GetInterpreter()->GetSubgraphIdx(model_id, device_flag);
      to_execute.device_id = device_idx;
      to_execute.sched_id = sched_id_++;

      EnqueueToWorker(to_execute);
    }
    lock.unlock();
  }
}

bool FixedDevicePlanner::NeedProfile() {
  return false;
}

}  // namespace impl
}  // namespace tflite
