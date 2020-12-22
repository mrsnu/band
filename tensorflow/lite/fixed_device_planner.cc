#include "tensorflow/lite/fixed_device_planner.h"
#include "tensorflow/lite/core/cpu/cpu.h"

namespace tflite {
namespace impl {

void FixedDevicePlanner::Plan() {
  auto cpuMask = tflite::impl::GetCPUThreadAffinityMask(
      static_cast<tflite::impl::TFLiteCPUMasks>(1));
  
  // SetCPUThreadAffinity(cpuMask);

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
      int device_idx = to_execute.device_id_ == -1 ?
                       model_id % GetInterpreter()->GetNumDevices() :
                       to_execute.device_id_;

      do {
        to_execute.subgraph_idx_ = GetInterpreter()->GetSubgraphIdx(
            model_id, static_cast<TfLiteDevice>(device_idx));
        to_execute.device_id_ = device_idx;
        device_idx = (device_idx + 1) % GetInterpreter()->GetNumDevices();
      } while(to_execute.subgraph_idx_ == -1);

      Worker& worker = GetInterpreter()->GetWorker(to_execute.device_id_);
      {
        std::lock_guard<std::mutex> lock(worker.GetDeviceMtx());
        worker.GetDeviceRequests().push_back(to_execute);
        /*
        if (to_execute.slo_ms_ == 0) {
          worker.GetDeviceRequests().push_back(to_execute);
        } else {
          worker.GetDeviceRequests().push_front(to_execute);
        }*/

        worker.GetRequestCv().notify_one();
      }
    }
    lock.unlock();
  }
}

}  // namespace impl
}  // namespace tflite
