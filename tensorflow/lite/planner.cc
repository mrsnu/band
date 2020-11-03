#include "tensorflow/lite/planner.h"
#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace impl {

Planner::Planner(Interpreter* interpreter)
  : planner_thread_([this]{this->Plan();}) {
  interpreter_ = interpreter;
  log_file_.open("/data/local/tmp/model_execution_log.csv", std::fstream::app);
  log_file_ << "model_id\tdevice_id\tenqueue_time\tinvoke_time\tend_time\n";
  log_file_.close();
}

Planner::~Planner() {
  // log_file_.close();
  planner_safe_bool_.terminate();
  planner_thread_.join();
}

TfLiteStatus Planner::Wait(int num_requests) {
  std::unique_lock<std::mutex> lock(job_queue_mtx_);
  end_invoke_.wait(lock, [this, num_requests]{
    return jobs_finished_.size() >= num_requests;
  });

  for (int i = 0; i < num_requests; ++i) {
    Job job = jobs_finished_.front();
    jobs_finished_.pop_front();
    log_file_.open("/data/local/tmp/model_execution_log.csv", std::fstream::app);
    log_file_ << job.model_id_ << "\t" << job.device_id_ << "\t" << job.enqueue_time_ << "\t" << job.invoke_time_ << "\t" << job.end_time_ << "\n";
    log_file_.close();
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
  job.enqueue_time_ = profiling::time::NowMicros();
  requests_.push_back(job);
  lock.unlock();

  planner_safe_bool_.notify();
}

}  // namespace impl
}  // namespace tflite
