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
                   << " cores for " 
                   << config.num_threads[device_flag_]
                   << " threads";

  const CpuSet worker_mask_set =
    TfLiteCPUMaskGetSet(config.cpu_masks[device_flag_]);
  return UpdateWorkerThread(worker_mask_set, config.num_threads[device_flag_]);
}

TfLiteStatus Worker::UpdateWorkerThread(const CpuSet thread_affinity_mask, int num_threads) {
  if (thread_affinity_mask.NumEnabled() == 0) {
    return kTfLiteError;
  }

  std::lock_guard<std::mutex> cpu_lock(cpu_mtx_);
  
  if (num_threads_ != num_threads) {
    num_threads_ = num_threads;
    need_cpu_update_ = true;
  }

  for (int cpu = 0; cpu < GetCPUCount(); cpu++) {
    if (cpu_set_.IsEnabled(cpu) != thread_affinity_mask.IsEnabled(cpu)) {
      cpu_set_ = thread_affinity_mask;
      need_cpu_update_ = true;
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

const CpuSet& Worker::GetWorkerThreadAffinity() const {
  return cpu_set_;
}

int Worker::GetNumThreads() const {
  return num_threads_;
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

TfLiteStatus Worker::TryCopyInputTensors(const Job& job) {
  // Skip all tensor communication for compute only case.
  if (job.input_handle < 0) {
    return kTfLiteOk;
  }

  Interpreter* interpreter = planner_.lock()->GetInterpreter();
  Subgraph* subgraph = interpreter->subgraph(job.subgraph_idx);
  std::bitset<TensorSize> unresolved_tensors = subgraph->inputs_mask();

  // Intermediate tensor communication
  for (int preceded_subgraph_index : job.previous_subgraph_indices) {
    Subgraph* preceded_subgraph = interpreter->subgraph(preceded_subgraph_index);
    const std::bitset<TensorSize>& output_mask = preceded_subgraph->outputs_mask();
    std::bitset<TensorSize> resolvable_tensors = unresolved_tensors & output_mask;
    if (resolvable_tensors.any()) {
      for (int tensor_index = 0; tensor_index < subgraph->tensors_size();
           tensor_index++) {
        if (resolvable_tensors.test(tensor_index)) {
          const TfLiteTensor* src = preceded_subgraph->tensor(tensor_index);
          TfLiteTensor* dst = subgraph->tensor(tensor_index);

          if (TfLiteTensorDataCopy(src, dst) == kTfLiteError) {
            TFLITE_LOG(ERROR)
                << "Tensor data copy failure. src name : " << src->name
                << ", dst name : " << dst->name;
            return kTfLiteError;
          }

          unresolved_tensors.set(tensor_index, 0);
        }
      }
    }
  }

  auto input_buffer = interpreter->model_input_buffer_[job.model_id].get();

  if (!input_buffer) {
    TFLITE_LOG(ERROR) << "No input buffer for model id " << job.model_id;
    return kTfLiteError;
  }

  // Copy model input
  for (int tensor_index = 0; tensor_index < subgraph->tensors_size(); tensor_index++) {
    if (unresolved_tensors.test(tensor_index)) {
      if (input_buffer->IsTensorIndexValid(tensor_index)) {
        if (input_buffer->GetTensorFromHandle(subgraph->tensor(tensor_index),
                                              tensor_index,
                                              job.input_handle) != kTfLiteOk) {
          return kTfLiteError;
        }
        unresolved_tensors.set(tensor_index, 0);
      } else {
        TFLITE_LOG(ERROR) << "Unresolved input tensor " << tensor_index
                          << " of subgraph " << job.subgraph_idx;
      }
    }
  }

  if (!unresolved_tensors.any()) {
    return kTfLiteOk;
  } else {
    return kTfLiteError;
  }
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

TfLiteStatus Worker::TryUpdateWorkerThread() {
  std::lock_guard<std::mutex> cpu_lock(cpu_mtx_);
  if (need_cpu_update_) {
    need_cpu_update_ = false;

    std::shared_ptr<Planner> planner_ptr = planner_.lock();
    Interpreter* interpreter_ptr = planner_ptr->GetInterpreter();
    auto internal_backend =
        interpreter_ptr->GetCpuBackendContext()->internal_backend_context();
    internal_backend->SetCpuSet(std::this_thread::get_id(), cpu_set_);
    internal_backend->SetMaxNumThreads(num_threads_);

    if (SetCPUThreadAffinity(cpu_set_) != kTfLiteOk) {
      TFLITE_LOG(ERROR) << "Worker " << device_flag_
                        << " failed to set cpu thread affinity";
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

}  // namespace impl
}  // namespace tflite
