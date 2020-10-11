#include "tensorflow/lite/planner.h"

namespace tflite {
namespace impl {

Planner::Planner(Interpreter* interpreter)
  : planner_thread_([this]{this->Plan();}) {
  interpreter_ = interpreter;
}

Planner::~Planner() {
  planner_safe_bool_.terminate();
  planner_thread_.join();
}

TfLiteStatus Planner::Wait(int num_requests) {
  std::unique_lock<std::mutex> lock(job_queue_mtx_);
  end_invoke_.wait(lock, [this, num_requests]{
    return jobs_finished_.size() >= num_requests;
  });

  for (int i = 0; i < num_requests; ++i) {
    jobs_finished_.pop_front();
  }
  lock.unlock();

  return kTfLiteOk;
}

void Planner::EnqueueFinishedJob(Job job) {
  std::unique_lock<std::mutex> lock(job_queue_mtx_);
  jobs_finished_.push_back(job);
  lock.unlock();

  end_invoke_.notify_one();
}

void Planner::EnqueueRequest(Job job) {
  std::unique_lock<std::mutex> lock(requests_mtx_);
  requests_.push_back(job);
  lock.unlock();

  planner_safe_bool_.notify();
}

}  // namespace impl
}  // namespace tflite
