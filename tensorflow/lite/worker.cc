#include "tensorflow/lite/worker.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"

namespace tflite {
namespace impl {

Worker::Worker(std::shared_ptr<Planner> planner)
  : device_cpu_thread_([this] { this->Work(); }) {
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

void Worker::Work() {
  // 0: All
  // 1: Little
  // 2: Big
  int status = set_cpu_thread_affinity(get_cpu_thread_affinity_mask(2));
  if (status != 0)
    return;

  while (true) {
    std::unique_lock<std::mutex> lock(device_mtx_);
    request_cv_.wait(lock, [this]() {
      return kill_worker_ || !this->requests_.empty();
    });

    if (requests_.empty()) {
      lock.unlock();
      break;
    }

    Job job = requests_.front();
    requests_.pop_front();
    lock.unlock();

    int subgraph_idx = job.subgraph_idx_;
    std::shared_ptr<Planner> planner_ptr = planner_.lock();
    if (planner_ptr) {
      Interpreter* interpreter_ptr = planner_ptr->GetInterpreter();
      Subgraph& subgraph = *(interpreter_ptr->subgraph(subgraph_idx));

      if (subgraph.Invoke() == kTfLiteOk) {
        planner_ptr->EnqueueFinishedJob(job);
      } else {
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
