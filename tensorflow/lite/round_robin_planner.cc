#include "tensorflow/lite/round_robin_planner.h"

#include <iostream>

namespace tflite {
namespace impl {

void RoundRobinPlanner::Plan() {
  while (true) {
    if (GetSafeBool().wait())
      return;

    std::vector<bool> is_empty;
    int device_idx = -1;
    for (int i = 0; i < GetInterpreter()->GetWorkersSize(); ++i) {
      Worker& worker = GetInterpreter()->GetWorker(i);
      {
        std::lock_guard<std::mutex> lock(worker.GetDeviceMtx());
        if (worker.GetDeviceRequests().empty()) {
          is_empty.push_back(true);
          if (device_idx == -1)
            device_idx = i;
        }
        else {
          is_empty.push_back(false);
        }
      }
    }

    std::unique_lock<std::mutex> lock(GetRequestsMtx());
    while (!GetRequests().empty() && device_idx != -1) {
      Job to_execute = Job(-1);
      to_execute = GetRequests().front();
      GetRequests().pop_front();

      if (to_execute.model_id_ != -1) {
        TfLiteDevice device_ = static_cast<TfLiteDevice>(device_idx);
        int subgraph_idx = 
          GetInterpreter()->GetSubgraphIdx(to_execute.model_id_, device_);

        if (subgraph_idx == -1) {
          std::cout << "no available m " << to_execute.model_id_ << std::endl;
          std::cout << "no available d " << device_ << std::endl;
          GetRequests().push_front(to_execute);
          break;
        } else {
          to_execute.subgraph_idx_ = subgraph_idx;
          to_execute.device_id_ = device_idx;
        }

        Worker& worker = GetInterpreter()->GetWorker(device_idx);
        {
          std::lock_guard<std::mutex> lock(worker.GetDeviceMtx());
          worker.GetDeviceRequests().push_back(to_execute);
          worker.GetRequestCv().notify_one();
        }

        is_empty[device_idx] = false;
        device_idx = -1;
        for (int i = 0; i < is_empty.size(); ++i) {
          if (is_empty[i]) {
            device_idx = i;
            break;
          }
        }
      }
      else {
        break;
      }
    }
    lock.unlock();

  }
}

}  // namespace impl
}  // namespace tflite
