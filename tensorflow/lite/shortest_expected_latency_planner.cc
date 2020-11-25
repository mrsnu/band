#include "tensorflow/lite/shortest_expected_latency_planner.h"
#include "tensorflow/lite/profiling/time.h"
#include <iostream>

namespace tflite {
namespace impl {

void ShortestExpectedLatencyPlanner::Plan() {
  while (true) {
    if (GetSafeBool().wait())
      return;

    std::deque<Job> local_jobs;
    // std::cout << "Before Acquire Lock" << std::endl;
    std::unique_lock<std::mutex> request_lock(GetRequestsMtx());
    if (!GetRequests().empty()) {
      GetRequests().swap(local_jobs);
      // to_execute = GetRequests().front();
      // GetRequests().pop_front();
    } else {
      continue;
    }
    request_lock.unlock();
    // std::cout << "After Release Lock" << std::endl;

    for (Job& to_execute : local_jobs) {
    // if (to_execute.model_id_ != -1) {
      // for (std::deque<Job>::iterator it = GetRequests().begin(); it != GetRequests().end(); ++it) {
      /*
      bool continue_plan = false;
      for (int i = 0; i < is_worker_empty.size(); ++i) {
        continue_plan |= is_worker_empty[i];
      }

      if (!continue_plan) {
        break;
      }*/

      // Job to_execute = *it;
      int model_id = to_execute.model_id_;
      TfLiteDevice device = GetInterpreter()->GetShortestLatency(model_id, to_execute);

      // if (is_worker_empty[device]) {
      Worker& worker = GetInterpreter()->GetWorker(device);
      {
        // std::cout << "Worker Lock" << std::endl;
        std::lock_guard<std::mutex> lock(worker.GetDeviceMtx());
        int subgraph_idx = GetInterpreter()->GetSubgraphIdx(model_id, device);
        to_execute.subgraph_idx_ = subgraph_idx;
        to_execute.device_id_ = device;
        // GetRequests().erase(it);

        worker.GetDeviceRequests().push_back(to_execute);
        worker.GetRequestCv().notify_one();

        // is_worker_empty[device] = false;
        // std::cout << "Worker Lock Release" << std::endl;
      }
      // }
    }
  }
}

}  // namespace impl
}  // namespace tflite
