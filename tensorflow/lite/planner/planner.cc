#include "tensorflow/lite/planner/planner.h"
#include "tensorflow/lite/profiling/time.h"
#include <fstream>

namespace tflite {
namespace impl {

Planner::Planner(Interpreter* interpreter) {
  interpreter_ = interpreter;
}

Planner::~Planner() {
  planner_safe_bool_.terminate();
  planner_thread_.join();
}

TfLiteStatus Planner::PrepareLogging(std::string log_path) {
  log_path_ = log_path;
  // Open file to write per-request timestamps later
  // NOTE: Columns starting `sched_id` are added for debugging purpose
  // and the metrics are only for ShortestExpectedLatency Planner.
  std::ofstream log_file(log_path_);
  if (!log_file.is_open())
    return kTfLiteError;
  log_file << "model_name\t"
           << "model_id\t"
           << "device_id\t"
           << "sched_id\t"
           << "enqueue_time\t"
           << "invoke_time\t"
           << "end_time\t"
           << "waiting_CPU\t"
           << "waiting_GPU\t"
           << "waiting_DSP\t"
           << "waiting_NPU\t"
           << "profiled_CPU\t"
           << "profiled_GPU\t"
           << "profiled_DSP\t"
           << "profiled_NPU\n";
  log_file.close();
  
  return kTfLiteOk;
}

void Planner::Wait() {
  std::unique_lock<std::mutex> lock(job_queue_mtx_);
  end_invoke_.wait(lock, [this]{
    return jobs_finished_.size() >= num_submitted_jobs_;
  });

  std::ofstream log_file(log_path_, std::ofstream::app);
  while (!jobs_finished_.empty()) {
    Job job = jobs_finished_.front();
    jobs_finished_.pop_front();

    // write all timestamp statistics to log file
    log_file << job.model_fname_ << "\t"
             << job.model_id_ << "\t"
             << job.device_id_ << "\t"
             << job.sched_id_ << "\t"
             << job.enqueue_time_ << "\t"
             << job.invoke_time_ << "\t"
             << job.end_time_ << "\t"
             << job.waiting_time[0] << "\t"
             << job.waiting_time[1] << "\t"
             << job.waiting_time[2] << "\t"
             << job.waiting_time[3] << "\t"
             << job.profiled_latency[0] << "\t"
             << job.profiled_latency[1] << "\t"
             << job.profiled_latency[2] << "\t"
             << job.profiled_latency[3] << "\n";
  }
  log_file.close();
  lock.unlock();
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
  num_submitted_jobs_++;
  lock.unlock();

  planner_safe_bool_.notify();
}

void Planner::EnqueueBatch(std::list<Job> jobs) {
  std::unique_lock<std::mutex> lock(requests_mtx_);
  auto enqueue_time = profiling::time::NowMicros();
  for (Job job : jobs) {
    job.enqueue_time_ = enqueue_time;
    requests_.push_back(job);
    num_submitted_jobs_++;
  }
  lock.unlock();

  planner_safe_bool_.notify();
}

void Planner::SetWindowSize(int schedule_window_size) {
  schedule_window_size_ = schedule_window_size;
}

}  // namespace impl
}  // namespace tflite
