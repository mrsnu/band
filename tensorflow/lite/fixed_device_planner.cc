#include "tensorflow/lite/fixed_device_planner.h"

namespace tflite {
namespace impl {

void FixedDevicePlanner::Plan() {
  while (true) {
    if (GetSafeBool().wait())
      return;

    // The lock will not be released until the request queue is empty,
    // which means concurrent enqueue is not available.
    // This can affect the performance.
    std::unique_lock<std::mutex> lock(GetRequestsMtx());
    while (!GetRequests().empty()) {
      Job to_execute = GetRequests().front();
      GetRequests().pop_front();

      int model_id = to_execute.model_id_;
      int device_idx = model_id % GetInterpreter()->GetNumDevices();
      to_execute.subgraph_idx_ = GetInterpreter()->GetSubgraphIdx(
          model_id, static_cast<TfLiteDevice>(device_idx));

      Worker& worker = GetInterpreter()->GetWorker(device_idx);
      {
        std::lock_guard<std::mutex> lock(worker.GetDeviceMtx());
        worker.GetDeviceRequests().push_back(to_execute);
        worker.GetRequestCv().notify_one();
      }
    }
    lock.unlock();
  }
}

}  // namespace impl
}  // namespace tflite
