#include "tensorflow/lite/worker.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"

namespace tflite {
namespace impl {

  Worker::Worker(Interpreter* interpreter, Planner* planner)
    : device_cpu_thread_([this] { this->Work(); }) {
    interpreter_ = interpreter;
    planner_ = planner;
  }

  Worker::~Worker() {
    request_cv_.notify_all();
    device_cpu_thread_.join();
  }

  void Worker::Work() {
    while (true) {
      std::unique_lock<std::mutex> lock(device_mtx_);
      request_cv_.wait(lock, [this]() {
        return planner_->GetKillWorkers() || !this->requests_.empty();
      });

      if (requests_.empty()) {
        lock.unlock();
        break;
      }

      Job job = requests_.front();
      requests_.pop_front();
      lock.unlock();

      int subgraph_idx = job.subgraph_idx_;
      Subgraph& subgraph = *(interpreter_->subgraph(subgraph_idx));

      if (subgraph.Invoke() == kTfLiteOk) {
        planner_->EnqueueFinishedJob(job);
      } else {
        break;
      }
    }
  }
}  // namespace impl
}  // namespace tflite
