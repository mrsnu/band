#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/worker.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
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

TfLiteStatus Worker::Init(WorkerConfig& config, int worker_id) {
  if (config.allow_worksteal) {
    AllowWorkSteal();
  }
  
  availability_check_interval_ms_ = config.availability_check_interval_ms;

  TFLITE_LOG_INTERNAL(TFLITE_LOG_INFO,
                      "Set affinity of %s to %s cores for %d threads.",
                      TfLiteDeviceGetName(device_flag_),
                      TfLiteCPUMaskGetName(config.cpu_masks[worker_id]),
                      config.num_threads[worker_id]);

  const CpuSet worker_mask_set =
    TfLiteCPUMaskGetSet(config.cpu_masks[worker_id]);
  return UpdateWorkerThread(worker_mask_set, config.num_threads[worker_id]);
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
    TFLITE_LOG_INTERNAL(TFLITE_LOG_INFO, "Availability check at %d ms.",
                        tflite::profiling::time::NowMicros());
    if (subgraph.Invoke() == kTfLiteOk) {
      return;
    }
  }
}

bool Worker::IsAvailable() {
  return !is_throttling_ && !is_paused_;
}

void Worker::Pause() {
  std::lock_guard<std::mutex> lock(device_mtx_);
  is_paused_ = true;
}

void Worker::Resume() {
  std::unique_lock<std::mutex> lock(device_mtx_);
  is_paused_ = false;
  lock.unlock();

  request_cv_.notify_one();
}

const CpuSet& Worker::GetWorkerThreadAffinity() const {
  return cpu_set_;
}

int Worker::GetNumThreads() const {
  return num_threads_;
}

JobQueue& Worker::GetDeviceRequests() {
  TFLITE_LOG_INTERNAL(TFLITE_LOG_WARNING,
                      "GetDeviceRequests() Not implemented");
  return requests_;
}

void Worker::AllowWorkSteal() {
  TFLITE_LOG_INTERNAL(TFLITE_LOG_WARNING, "AllowWorkSteal() Not implemented");
}

bool Worker::IsBusy() {
  TFLITE_LOG_INTERNAL(TFLITE_LOG_WARNING, "IsBusy() Not implemented");
  return false;
}

ErrorReporter* Worker::GetErrorReporter() {
  // TODO(dostos): thread-safety for error reporter
  auto planner_ptr = planner_.lock();
  if (planner_ptr) {
    return planner_ptr->GetInterpreter()->GetErrorReporter();
  } else {
    nullptr;
  }
}

TfLiteStatus Worker::TryCopyInputTensors(const Job& job) {
  // Skip all tensor communication for compute only case.
  if (job.input_handle < 0) {
    return kTfLiteOk;
  }

  Interpreter* interpreter = planner_.lock()->GetInterpreter();
  Subgraph* subgraph = interpreter->subgraph(job.subgraph_idx);
  std::set<int> unresolved_tensors(subgraph->inputs().begin(), subgraph->inputs().end());
  // Intermediate tensor communication
  for (auto subgraph_it = job.previous_subgraph_indices.cbegin(); subgraph_it != job.previous_subgraph_indices.cend(); ++subgraph_it) {
    int preceded_subgraph_index = *subgraph_it;
    Subgraph* preceded_subgraph = interpreter->subgraph(preceded_subgraph_index);

    for(int tensor_index: preceded_subgraph->outputs()) {
      if (unresolved_tensors.find(tensor_index) != unresolved_tensors.end()) {
        const TfLiteTensor* src = preceded_subgraph->tensor(tensor_index);
        TfLiteTensor* dst = subgraph->tensor(tensor_index);

        if (TfLiteTensorDataCopy(src, dst) == kTfLiteError) {
          TF_LITE_MAYBE_REPORT_ERROR(
              GetErrorReporter(),
              "Tensor data copy failure from %s to %s", src->name,
              dst->name);
          return kTfLiteError;
        }

        unresolved_tensors.erase(tensor_index);
      }
    }
  }

  auto input_buffer = interpreter->model_input_buffer_[job.model_id].get();

  if (!input_buffer) {
    TF_LITE_MAYBE_REPORT_ERROR(GetErrorReporter(),
                               "No input buffer for model id %d",
                               job.model_id);
    return kTfLiteError;
  }

  // Copy model input
  for (auto tensor_it = unresolved_tensors.begin(); tensor_it != unresolved_tensors.end();) {
    int tensor_index = *tensor_it;
    if (input_buffer->IsTensorIndexValid(tensor_index)) {
      if (input_buffer->GetTensorFromHandle(subgraph->tensor(tensor_index),
                                            tensor_index,
                                            job.input_handle) != kTfLiteOk) {
        return kTfLiteError;
      }
      tensor_it = unresolved_tensors.erase(tensor_it);
    } else {
      TF_LITE_MAYBE_REPORT_ERROR(
          GetErrorReporter(),
          "Unresolved input tensor %d of subgraph %d", tensor_index,
          job.subgraph_idx);
      ++tensor_it;
    }
  }

  if (unresolved_tensors.empty()) {
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
    TF_LITE_MAYBE_REPORT_ERROR(GetErrorReporter(),
                               "No output buffer for model id %d",
                               job.model_id);
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
      TF_LITE_MAYBE_REPORT_ERROR(
          GetErrorReporter(),
          "Worker for device %d failed to set cpu thread affinity",
          device_flag_);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

}  // namespace impl
}  // namespace tflite
