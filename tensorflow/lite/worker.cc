#include "tensorflow/lite/worker.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include <iostream>

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

        for (auto tensor_idx : subgraph.outputs()) {
          TfLiteTensor& tensor = subgraph.context()->tensors[tensor_idx]; 

          std::cout << "Tensor Size (bytes) : " << tensor.bytes << std::endl;

          std::cout << "Tensor Dimension" << std::endl;
          for (int j = 0; j < tensors.dims->size; ++j) {
            int dim = tensor.dims->data[j];
            std::cout << dim << " ";
          }
          std::cout << std::endl;

          std::cout << "Tensor Data - " << std::endl;
          for (int j = 0; j < bytes; ++j) {
            // 1.
            // Used `char` under the assumption that the output data type
            // to be INT8. (`tensor.data.raw` is also char*.)
            // Refer to lite/c/common.h:TfLiteTensor, TfLitePtrUnion, for more
            // details on `tensor.data` and `tensor.data.raw`
            // 2.
            // Memory layout is unclear.
            char value = *(tensor.data.raw + j);
            std::cout << (int)value << " ";
          }
          std::cout << std::endl;
        }

        planner_ptr->EnqueueFinishedJob(job);
      } else {
        // TODO #21: Handle errors in multi-thread environment
        // Currently, put a job with a minus sign if Invoke() fails.
        job.end_time_ = profiling::time::NowMicros();
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

  int64_t total = 0;
  for (std::deque<Job>::iterator it = requests_.begin(); it != requests_.end(); ++it) {
    int subgraph_idx = (*it).subgraph_idx_;
    std::shared_ptr<Planner> planner_ptr = planner_.lock();
    if (planner_ptr) {
      Interpreter* interpreter_ptr = planner_ptr->GetInterpreter();
      Subgraph& subgraph = *(interpreter_ptr->subgraph(subgraph_idx));
      int64_t subgraph_latency = subgraph.GetExpectedLatency();
      total += subgraph_latency;

      if (it == requests_.begin()) {
        int64_t current_time = profiling::time::NowMicros();
        int64_t invoke_time = (*it).invoke_time_;
        if (invoke_time > 0 && current_time > invoke_time) {
          int64_t progress = (current_time - invoke_time) > subgraph_latency ? subgraph_latency : (current_time - invoke_time);
          total -= progress;
          // std::cout << "Invoke Time : " << (*it).invoke_time_ << std::endl;
          // std::cout << "current Time : " << current_time << std::endl;
          // std::cout << "subgraph_latency : " << subgraph_latency << std::endl;
          // std::cout << "progress : " << progress << std::endl;
        }
      }
    } else {
      return -1;
    }
  }
  lock.unlock();

  return total;
}

}  // namespace impl
}  // namespace tflite
