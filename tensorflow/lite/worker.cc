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

TfLiteStatus Worker::CopyInputTensors(const Job& job) {
  // Compute only.
  if (job.input_handle < 0) {
    return kTfLiteOk;
  }

  Interpreter* interpreter = planner_.lock()->GetInterpreter();
  Subgraph* subgraph = interpreter->subgraph(job.subgraph_idx);
  auto input_buffer = interpreter->model_input_buffer[job.model_id].get();
  
  if (!input_buffer) {
    TFLITE_LOG(ERROR) << "No input buffer for model id " << job.model_id;
    return kTfLiteError;
  }

  Tensors input_tensors;
  if (input_buffer->Get(input_tensors, job.input_handle) == kTfLiteOk) {
    auto input_indices = subgraph->inputs();
    for (size_t i = 0; i < input_indices.size(); i++) {
      if (TfLiteTensorDataCopy(subgraph->tensor(input_indices[i]),
                               input_tensors[i]) != kTfLiteOk) {
        TFLITE_LOG(ERROR) << "Input copy failure.";
      }
    }
    return kTfLiteOk;
  } else {
    TFLITE_LOG(ERROR) << "Input tensors are null model " << job.model_id
                      << " input handle " << job.input_handle;
    return kTfLiteError;
  }
}

TfLiteStatus Worker::CopyOutputTensors(const Job& job) {
  // Compute only.
  if (job.output_handle < 0 || !job.is_final_subgraph) {
    return kTfLiteOk;
  }

  Interpreter* interpreter = planner_.lock()->GetInterpreter();
  Subgraph* subgraph = interpreter->subgraph(job.subgraph_idx);
  auto output_buffer = interpreter->model_output_buffer[job.model_id].get();
  
  if (!output_buffer) {
    TFLITE_LOG(ERROR) << "No output buffer for model id " << job.model_id;
    return kTfLiteError;
  }

  auto output_indices = subgraph->outputs();
  std::vector<TfLiteTensor*> output_tensors(output_indices.size());
  for (size_t i = 0; i < output_indices.size(); i++) {
    output_tensors[i] = subgraph->tensor(output_indices[i]);
  }

  return output_buffer->Put(output_tensors, job.output_handle);
}

TfLiteStatus Worker::ProcessJob(Job& job, std::function<void(Job&)> pre_process, std::function<void(Job&)> pre_invoke,
                                std::function<void(Job&)> post_invoke, std::function<void(Job&)> post_process) {
  std::shared_ptr<Planner> planner = planner_.lock();
  if (!planner) {
    return kTfLiteError;
  }
  Interpreter* interpreter = planner->GetInterpreter();
  Subgraph* subgraph = interpreter->subgraph(job.subgraph_idx);
  pre_process(job);
  if (CopyInputTensors(job) == kTfLiteOk) {
    pre_invoke(job);

    if (subgraph->Invoke() == kTfLiteOk) {
      post_invoke(job);
      interpreter->UpdateProfileResult(subgraph->GetKey(),
                                       (job.end_time - job.invoke_time));
      // TODO #65: Tensor communications between subgraphs
      interpreter->InvokeModelsAsync(job.following_jobs);
      if (CopyOutputTensors(job) == kTfLiteOk) {
        job.status = kTfLiteJobSuccess;
      } else {
        job.status = kTfLiteJobOutputCopyFailure;
      }
    } else {
      job.status = kTfLiteJobInvokeFailure;
    }
  } else {
    job.status = kTfLiteJobInputCopyFailure;
  }
  post_process(job);
  if (job.slo_us > 0 && job.is_final_subgraph && job.status == kTfLiteJobSuccess) {
    // check if slo has been violated or not
    auto latency = job.end_time - job.enqueue_time;
    if (latency > job.slo_us) {
      job.status = kTfLiteJobSLOViolation;
    }
  }
  planner->EnqueueFinishedJob(job);
  return job.status == kTfLiteJobSuccess ? kTfLiteOk : kTfLiteError;
}

}  // namespace impl
}  // namespace tflite
