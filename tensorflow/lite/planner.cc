#include "tensorflow/lite/planner.h"
#include "tensorflow/lite/profiling/time.h"
#include <fstream>

namespace tflite {
namespace impl {

Planner::Planner(Interpreter* interpreter) {
  interpreter_ = interpreter;

  // open file to write per-request timestamps later
  // TODO: make the file path a configurable command line arg
  std::ofstream log_file(log_path_);
  log_file << "job_id\t"
           << "model_name\t"
           << "model_id\t"
           << "device_id\t"
           << "enqueue_time\t"
           << "invoke_time\t"
           << "end_time\n";
  log_file.close();
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

  std::ofstream log_file(log_path_, std::ofstream::app);
  for (int i = 0; i < num_requests; ++i) {
    Job job = jobs_finished_.front();
    jobs_finished_.pop_front();

    // write all timestamp statistics to log file
    log_file << job.sched_id_ << "\t"
             << job.model_fname_ << "\t"
             << job.model_id_ << "\t"
             << job.device_id_ << "\t"
             << job.enqueue_time_ << "\t"
             << job.invoke_time_ << "\t"
             << job.end_time_ << "\n";
  }
  log_file.close();
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
  job.enqueue_time_ = profiling::time::NowMicros();
  std::unique_lock<std::mutex> lock(requests_mtx_);
  requests_.push_back(job);
  lock.unlock();

  planner_safe_bool_.notify();
}

void Planner::EnqueueBatch(std::list<Job> jobs) {
  std::unique_lock<std::mutex> lock(requests_mtx_);
  auto enqueue_time = profiling::time::NowMicros();
  for (Job job : jobs) {
    job.enqueue_time_ = enqueue_time;
    requests_.push_back(job);
  }
  lock.unlock();

  planner_safe_bool_.notify();
}

}  // namespace impl
}  // namespace tflite
