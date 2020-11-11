#include "tensorflow/lite/round_robin_planner.h"

namespace tflite {
namespace impl {

void RoundRobinPlanner::Plan() {
  while (true) {
    if (GetSafeBool().wait())
      return;

    int device_idx = -1;
    for (int i = 0; i < GetInterpreter()->GetWorkersSize(); ++i) {
      Worker& worker = GetInterpreter()->GetWorker(i);
      {
        std::lock_guard<std::mutex> lock(worker.GetDeviceMtx());
        if (worker.GetDeviceRequests().empty()) {
          device_idx = i;
          break;
        }
      }
    }

    if (device_idx != -1) {
      Job to_execute = Job(-1);
      std::unique_lock<std::mutex> lock(GetRequestsMtx());
      if (!GetRequests().empty()) {
        to_execute = GetRequests().front();
        GetRequests().pop_front();
      }
      lock.unlock();

      if (to_execute.model_id_ != -1) {
        TfLiteDevice device_ = static_cast<TfLiteDevice>(device_idx);
        to_execute.subgraph_idx_ =
          GetInterpreter()->GetSubgraphIdx(to_execute.model_id_, device_);
        to_execute.device_id_ = device_idx;

        Worker& worker = GetInterpreter()->GetWorker(device_idx);
        {
          std::lock_guard<std::mutex> lock(worker.GetDeviceMtx());
          worker.GetDeviceRequests().push_back(to_execute);
          worker.GetRequestCv().notify_one();
        }
      }
    }
  }
}

}  // namespace impl
}  // namespace tflite
