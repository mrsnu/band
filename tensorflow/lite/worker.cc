#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/worker.h"
#include "tensorflow/lite/core/subgraph.h"
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

TfLiteStatus Worker::TryCopyInputTensors(const Job& job) {
  // Compute only.
  if (job.input_handle < 0) {
    return kTfLiteOk;
  }

  Interpreter* interpreter = planner_.lock()->GetInterpreter();
  Subgraph* subgraph = interpreter->subgraph(job.subgraph_idx);
  auto input_buffer = interpreter->model_input_buffer[job.model_id].get();

  const std::vector<TfLiteTensor>* input_tensors = input_buffer->Get(job.input_handle);

  if (input_tensors) {
    auto input_indices = subgraph->inputs();
    for (size_t i = 0; i < input_indices.size(); i++) {
      std::memcpy(subgraph->tensor(i)->data.raw, input_tensors->at(i).data.raw,
                  input_tensors->at(i).bytes);
    }
    return kTfLiteOk;
  } else {
    TFLITE_LOG(ERROR) << "Input tensors are null model " << job.model_id << " input handle " << job.input_handle;
    return kTfLiteError;
  }
}

}  // namespace impl
}  // namespace tflite
