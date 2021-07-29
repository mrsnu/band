#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/worker.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/tools/logging.h"
#include "tensorflow/lite/profiling/time.h"

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
  availability_check_interval_ms_ = config.availability_check_interval_ms;

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

void Worker::WaitUntilDeviceAvailable(Subgraph& subgraph) {
  while (true) {
    tflite::profiling::time::SleepForMicros(1000 * availability_check_interval_ms_);
    TFLITE_LOG(INFO) << "Availability check at " << tflite::profiling::time::NowMicros();
    if (subgraph.Invoke() == kTfLiteOk) {
      return;
    }
  }
}

bool Worker::IsAvailable() {
  std::lock_guard<std::mutex> lock(device_mtx_);
  return is_available_;
}

JobQueue& Worker::GetDeviceRequests() {
  TFLITE_LOG(ERROR) << "Worker::GetDeviceRequests() Not implemented.";
  return requests_;
}

void Worker::AllowWorkSteal() {
  TFLITE_LOG(ERROR) << "Worker::AllowWorkSteal() Not implemented.";
}

bool Worker::IsBusy() {
  TFLITE_LOG(ERROR) << "Worker::IsBusy() Not implemented.";
  return false;
}

TfLiteStatus CopyTensors(Subgraph& src_subgraph, Subgraph& dst_subgraph) {
  TfLiteStatus ret = kTfLiteError;
  for (int output_index : src_subgraph.outputs()) {
    for (int input_index : dst_subgraph.inputs()) {
      if (output_index == input_index) {
         const TfLiteTensor* src = src_subgraph.tensor(output_index);
         TfLiteTensor* dst = dst_subgraph.tensor(input_index);

         if (TfLiteTensorDataCopy(src, dst) == kTfLiteError) {
           TFLITE_LOG(ERROR)
               << "Tensor data copy failure. src name : " << src->name
               << ", dst name : " << dst->name;
           return kTfLiteError;
         }
         ret = kTfLiteOk;
      }
    }
  }
  return ret;
}

TfLiteStatus Worker::TryCopyInputTensors(const Job& job) {
  // Compute only.
  if (job.input_handle < 0) {
    return kTfLiteOk;
  }

  Interpreter* interpreter = planner_.lock()->GetInterpreter();
  Subgraph* subgraph = interpreter->subgraph(job.subgraph_idx);

  if (job.previous_subgraph_idx != -1) {
    Subgraph* prev_subgraph = interpreter->subgraph(job.previous_subgraph_idx);
    return CopyTensors(*prev_subgraph, *subgraph);
  }

  auto input_buffer = interpreter->model_input_buffer_[job.model_id].get();
  
  if (!input_buffer) {
    TFLITE_LOG(ERROR) << "No input buffer for model id " << job.model_id;
    return kTfLiteError;
  }

  for (int subgraph_input : subgraph->inputs()) {
    if (input_buffer->IsTensorIndexValid(subgraph_input)) {
      if (input_buffer->GetTensorFromHandle(subgraph->tensor(subgraph_input),
                                            subgraph_input,
                                            job.input_handle) != kTfLiteOk) {
        return kTfLiteError;
      }
    }
  }

  return kTfLiteOk;
}

TfLiteStatus Worker::TryCopyOutputTensors(const Job& job) {
  // Compute only.
  if (job.output_handle < 0) {
    return kTfLiteOk;
  }

  Interpreter* interpreter = planner_.lock()->GetInterpreter();
  auto output_buffer = interpreter->model_output_buffer_[job.model_id].get();

  if (!output_buffer) {
    TFLITE_LOG(ERROR) << "No output buffer for model id " << job.model_id;
    return kTfLiteError;
  }

  Subgraph* subgraph = interpreter->subgraph(job.subgraph_idx);

  for (int subgraph_output : subgraph->outputs()) {
    if (output_buffer->IsTensorIndexValid(subgraph_output)) {
      if (output_buffer->PutTensorToHandle(subgraph->tensor(subgraph_output),
                                           subgraph_output,
                                           job.output_handle) != kTfLiteOk) {
        return kTfLiteError;
      }
    }
  }

  return kTfLiteOk;
}

bool Worker::IsValid(Job& job) {
  return job.model_id >= 0
      && job.subgraph_idx >= 0
      && job.device_id >= 0
      && job.enqueue_time > 0
      && job.invoke_time == 0
      && job.end_time == 0;
}

void Worker::PrepareReenqueue(Job& job, Planner* planner) {
  job.invoke_time = 0;
  job.end_time = 0;
  job.resolved_tensors =
      planner->GetInterpreter()->GetModelSpec(job.model_id).input_tensors;
}

}  // namespace impl
}  // namespace tflite
