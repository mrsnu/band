#include "tensorflow/lite/round_robin_planner.h"

namespace tflite {
namespace impl {

void RoundRobinPlanner::Plan() {
  while (true) {
    if (GetSafeBool().wait())
      return;

    std::unique_lock<std::mutex> lock(GetRequestsMtx());
    while (!GetRequests().empty()) {
      Job to_execute = GetRequests().front();

      int device_idx = -1;
      for (int i = 0; i < GetInterpreter()->GetWorkersSize(); ++i) {
        Worker& worker = GetInterpreter()->GetWorker(i);
        {
          std::lock_guard<std::mutex> lock(worker.GetDeviceMtx());
          if (worker.GetDeviceRequests().empty()) {
            device_idx = i;

            TfLiteDevice device_ = static_cast<TfLiteDevice>(device_idx);
            if (device_ == kTfLiteTPU && to_execute.model_id_ == 2)
              continue;
            to_execute.subgraph_idx_ =
              GetInterpreter()->GetSubgraphIdx(to_execute.model_id_, device_);
            to_execute.device_id_ = device_idx;

            GetRequests().pop_front();

            worker.GetDeviceRequests().push_back(to_execute);
            worker.GetRequestCv().notify_one();
            break;
          }
        }
      }

      if (device_idx == -1)
        break;
    }
    lock.unlock();
  }
}

}  // namespace impl
}  // namespace tflite
