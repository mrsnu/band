#include "tensorflow/lite/worker.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/time.h"

#include <atomic>

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

    Job job = requests_.front();
    requests_.pop_front();
    lock.unlock();

    int subgraph_idx = job.subgraph_idx_;
    std::shared_ptr<Planner> planner_ptr = planner_.lock();
    if (planner_ptr) {
      Interpreter* interpreter_ptr = planner_ptr->GetInterpreter();
      Subgraph& subgraph = *(interpreter_ptr->subgraph(subgraph_idx));

      int64_t average_freq = 0;
      std::atomic_bool watcher_loop = ATOMIC_VAR_INIT(true);
      CpuSet cpu_set;
      GetCPUThreadAffinity(cpu_set);

      auto frequency_watcher = [&](){
        int64_t count = 0;
        while (watcher_loop.load(std::memory_order_acquire)) {
          average_freq += GetCPUAverageFrequencyKhz(cpu_set);
          ++count;
          std::this_thread::sleep_for(std::chrono::nanoseconds(100000));
        };

        average_freq /= count;
      };

      std::thread watcher_thread(frequency_watcher);

      job.invoke_time_ = profiling::time::NowMicros();

      if (subgraph.Invoke() == kTfLiteOk) {
        job.end_time_ = profiling::time::NowMicros();
        watcher_loop.store(false, std::memory_order_release);
        watcher_thread.join();
        job.average_freq_ = average_freq;
        planner_ptr->EnqueueFinishedJob(job);
      } else {
        job.end_time_ = profiling::time::NowMicros();
        watcher_loop.store(false, std::memory_order_release);
        watcher_thread.join();
        job.average_freq_ = average_freq;
        // TODO #21: Handle errors in multi-thread environment
        // Currently, put a job with a minus sign if Invoke() fails.
        planner_ptr->EnqueueFinishedJob(Job(-1 * subgraph_idx));
      }

      planner_ptr->GetSafeBool().notify();
    } else {
      // TODO #21: Handle errors in multi-thread environment
      return;
    }
  }
}

}  // namespace impl
}  // namespace tflite
