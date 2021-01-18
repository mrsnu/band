#include "tensorflow/lite/worker.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace impl {

Worker::Worker(std::shared_ptr<Planner> planner) {
  planner_ = planner;
  device_cpu_thread_ = std::thread([this]{ this->Work(); });
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

    std::unique_lock<std::mutex> cpu_lock(cpu_set_mtx_);
    if (need_cpu_set_update_) {
      need_cpu_set_update_ = false;
      if (SetCPUThreadAffinity(cpu_set_) != kTfLiteOk) {
        // TODO #21: Handle errors in multi-thread environment
        cpu_lock.unlock();
        break;
      }
    }
    cpu_lock.unlock();

    Job& job = requests_.front();
    // requests_.pop_front();
    lock.unlock();

    int subgraph_idx = job.subgraph_idx_;
    std::shared_ptr<Planner> planner_ptr = planner_.lock();
    if (planner_ptr) {
      Interpreter* interpreter_ptr = planner_ptr->GetInterpreter();
      Subgraph& subgraph = *(interpreter_ptr->subgraph(subgraph_idx));
      job.invoke_time_ = profiling::time::NowMicros();

      if (subgraph.Invoke() == kTfLiteOk) {
        job.end_time_ = profiling::time::NowMicros();
        planner_ptr->EnqueueFinishedJob(job);
      } else {
        job.end_time_ = profiling::time::NowMicros();
        // TODO #21: Handle errors in multi-thread environment
        // Currently, put a job with a minus sign if Invoke() fails.
        planner_ptr->EnqueueFinishedJob(Job(-1 * subgraph_idx));
      }

      lock.lock();
      requests_.pop_front();
      lock.unlock();

      planner_ptr->GetSafeBool().notify();
    } else {
      // TODO #21: Handle errors in multi-thread environment
      return;
    }
  }
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
    int model_id = (*it).model_id_;
    TfLiteDeviceFlags device_id =
        static_cast<TfLiteDeviceFlags>((*it).device_id_);
    int64_t profiled_latency =
        interpreter->GetProfiledLatency(model_id, device_id);

    total += profiled_latency;
    if (it == requests_.begin()) {
      int64_t current_time = profiling::time::NowMicros();
      int64_t invoke_time = (*it).invoke_time_;
      if (invoke_time > 0 && current_time > invoke_time) {
        int64_t progress = (current_time - invoke_time) > profiled_latency ? profiled_latency : (current_time - invoke_time);
        total -= progress;
      }
    }
  }
  lock.unlock();

  return total;
 }


}  // namespace impl
}  // namespace tflite
