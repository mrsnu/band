#include "tensorflow/lite/worker.h"

#include "tensorflow/lite/tools/logging.h"


namespace tflite {
namespace impl {

Worker::Worker(std::shared_ptr<Planner> planner, TfLiteDeviceFlags device_flag)
  : device_flag_(device_flag) {
  planner_ = planner;
}

Worker::~Worker() {
  {
    std::lock_guard<std::mutex> lock(device_mtx_);
    kill_worker_ = true;
  }
  request_cv_.notify_all();
  device_cpu_thread_.join();
}

TfLiteStatus Worker::Init(WorkerConfig& config) {
  if (config.allow_worksteal) {
    AllowWorkSteal();
  }

  TFLITE_LOG(INFO) << "Set affinity of "
                   << TfLiteDeviceGetName(device_flag_)
                   << " to "
                   << TfLiteCPUMaskGetName(config.cpu_masks[device_flag_])
                   << " cores";

  const CpuSet worker_mask_set =
    TfLiteCPUMaskGetSet(config.cpu_masks[device_flag_]);
  return SetWorkerThreadAffinity(worker_mask_set);
}

TfLiteStatus Worker::SetWorkerThreadAffinity(const CpuSet thread_affinity_mask) {
  if (thread_affinity_mask.NumEnabled() == 0) {
    return kTfLiteError;
  }

  std::unique_lock<std::mutex> cpu_lock(cpu_set_mtx_);
  for (int cpu = 0; cpu < GetCPUCount(); cpu++) {
    if (cpu_set_.IsEnabled(cpu) != thread_affinity_mask.IsEnabled(cpu)) {
      cpu_set_ = thread_affinity_mask;
      need_cpu_set_update_ = true;
      return kTfLiteOk;
    }
  }
  return kTfLiteOk;
}

std::deque<Job>& Worker::GetDeviceRequests() {
  TFLITE_LOG(ERROR) << "Worker::GetDeviceRequests() Not implemented.";
  return requests_;
}

void Worker::AllowWorkSteal() {
  TFLITE_LOG(ERROR) << "Worker::AllowWorkSteal() Not implemented.";
}

bool Worker::GiveJob(Job& job) {
  TFLITE_LOG(ERROR) << "Worker::GiveJob() Not implemented.";
  return false;
}

bool Worker::IsBusy() {
  TFLITE_LOG(ERROR) << "Worker::IsBusy() Not implemented.";
  return false;
}

}  // namespace impl
}  // namespace tflite
