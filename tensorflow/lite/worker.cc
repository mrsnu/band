#include "tensorflow/lite/worker.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace impl {

Worker::Worker(std::shared_ptr<Planner> planner, TfLiteDeviceFlags device_flag)
  : device_cpu_thread_([this] { this->Work(); }), device_flag_(device_flag) {
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
  if (thread_affinity_mask.NumEnabled() == 0)
    return kTfLiteError;
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

void Worker::Work() {
  while (true) {
    std::unique_lock<std::mutex> lock(device_mtx_);
    request_cv_.wait(lock, [this]() {
      return kill_worker_ || !this->requests_.empty();
    });

    if (requests_.empty()) {
      lock.unlock();
      break;
    }

    Job& job = requests_.front();
    lock.unlock();

    int subgraph_idx = job.subgraph_idx;
    std::shared_ptr<Planner> planner_ptr = planner_.lock();
    if (planner_ptr) {
      Interpreter* interpreter_ptr = planner_ptr->GetInterpreter();
      Subgraph& subgraph = *(interpreter_ptr->subgraph(subgraph_idx));

      { 
        std::lock_guard<std::mutex> cpu_lock(cpu_set_mtx_);
        if (need_cpu_set_update_) {
          need_cpu_set_update_ = false;

          auto internal_backend = interpreter_ptr->GetCpuBackendContext()
                                      ->internal_backend_context();
          internal_backend->SetCpuSet(std::this_thread::get_id(), cpu_set_);

          if (SetCPUThreadAffinity(cpu_set_) != kTfLiteOk) {
            // TODO #21: Handle errors in multi-thread environment
            break;
          }
        }
      }

      job.invoke_time = profiling::time::NowMicros();

      if (subgraph.Invoke() == kTfLiteOk) {
        job.end_time = profiling::time::NowMicros();
        // TODO #65: Tensor communications between subgraphs
        interpreter_ptr->InvokeModelsAsync(job.following_jobs);
        planner_ptr->EnqueueFinishedJob(job);
      } else {
        job.end_time = profiling::time::NowMicros();
        // TODO #21: Handle errors in multi-thread environment
        // Currently, put a job with a minus sign if Invoke() fails.
        planner_ptr->EnqueueFinishedJob(Job(-1 * subgraph_idx));
      }
      lock.lock();
      requests_.pop_front();
      bool empty = requests_.empty();
      lock.unlock();

      if (allow_work_steal_ && empty) {
        TryWorkSteal();
      }

      planner_ptr->GetSafeBool().notify();
    } else {
      // TODO #21: Handle errors in multi-thread environment
      return;
    }
  }
}

void Worker::TryWorkSteal() {
  std::shared_ptr<Planner> planner_ptr = planner_.lock();
  if (!planner_ptr) {
    TFLITE_LOG(ERROR) << "Worker " << device_flag_
                      << " TryWorkSteal() Failed to acquire pointer to Planner";
    return;
  }

  Interpreter* interpreter_ptr = planner_ptr->GetInterpreter();
  int64_t max_latency_gain = -1;
  int max_latency_gain_device = -1;
  for (auto& device_and_worker: interpreter_ptr->GetWorkers()) {
    TfLiteDeviceFlags target_device = device_and_worker.first;
    Worker* target_worker = device_and_worker.second.get();
    if (target_device == device_flag_) {
      continue;
    }

    int64_t waiting_time = target_worker->GetWaitingTime();

    std::unique_lock<std::mutex> lock(target_worker->GetDeviceMtx());
    if (target_worker->GetDeviceRequests().size() < 2) {
      // There is nothing to steal here or
      // the job is being processed by the target worker,
      // so leave it alone.
      continue;
    }

    Job& job = target_worker->GetDeviceRequests().back();
    lock.unlock();

    SubgraphKey key(job.model_id, device_flag_, job.start_idx, job.end_idx);
    int64_t expected_latency = interpreter_ptr->GetSubgraphProfileResult(key);
    if (expected_latency == -1 || expected_latency > waiting_time) {
      // no point in stealing this job, it's just going to take longer
      continue;
    }

    int64_t latency_gain = waiting_time - expected_latency;
    if (latency_gain > max_latency_gain) {
      max_latency_gain = latency_gain;
      max_latency_gain_device = target_device;
    }
  }

  if (max_latency_gain < 0) {
    // no viable job to steal -- do nothing
    return;
  }

  Worker* target_worker = interpreter_ptr->GetWorker(max_latency_gain_device);
  std::unique_lock<std::mutex> lock(target_worker->GetDeviceMtx(), std::defer_lock);
  std::unique_lock<std::mutex> my_lock(device_mtx_, std::defer_lock);
  std::lock(lock, my_lock);

  if (target_worker->GetDeviceRequests().empty()) {
    // target worker has went on and finished all of its jobs
    // while we were slacking off
    return;
  }

  // this must not be a reference,
  // otherwise the pop_back() below will invalidate it
  Job job = target_worker->GetDeviceRequests().back();
  if (job.invoke_time > 0) {
    // make sure the target worker hasn't started processing the job yet
    return;
  }

  if (!requests_.empty()) {
    // make sure that I still don't have any work to do
    return;
  }

  int subgraph_idx =
    interpreter_ptr->GetSubgraphIdx(job.model_id, device_flag_);
  job.subgraph_idx = subgraph_idx;
  job.device_id = device_flag_;

  // finally, we perform the swap
  target_worker->GetDeviceRequests().pop_back();
  requests_.push_back(job);
}

int64_t Worker::GetWaitingTime() {
  std::unique_lock<std::mutex> lock(device_mtx_);

  std::shared_ptr<Planner> planner = planner_.lock();
  if (!planner) {
    return -1;
  }
  Interpreter* interpreter = planner->GetInterpreter();

  int64_t total = 0;
  for (std::deque<Job>::iterator it = requests_.begin();
       it != requests_.end(); ++it) {
    int model_id = (*it).model_id;
    TfLiteDeviceFlags device_id =
        static_cast<TfLiteDeviceFlags>((*it).device_id);
    int start_idx = (*it).start_idx;
    int end_idx = (*it).end_idx;
    SubgraphKey key(model_id, device_id, start_idx, end_idx);
    int64_t profiled_latency = interpreter->GetSubgraphProfileResult(key);

    total += profiled_latency;
    if (it == requests_.begin()) {
      int64_t current_time = profiling::time::NowMicros();
      int64_t invoke_time = (*it).invoke_time;
      if (invoke_time > 0 && current_time > invoke_time) {
        int64_t progress =
          (current_time - invoke_time) > profiled_latency ? profiled_latency
                                              : (current_time - invoke_time);
        total -= progress;
      }
    }
  }
  lock.unlock();

  return total;
}


}  // namespace impl
}  // namespace tflite
