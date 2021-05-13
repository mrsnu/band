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
  log_file << "sched_id,"
           << "model_name,"
           << "model_id,"
           << "device_id,"
           << "start_idx,"
           << "end_idx,"
           << "subgraph_idx,"
           << "enqueue_time,"
           << "invoke_time,"
           << "end_time,"
           << "expected_execution_time,"
           << "actual_execution_time,"
           << "expected_latency,"
           << "actual_latency,"
           << "slo,"
           << "is_finished,"
           << "is_violated\n";
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
    log_file << job.sched_id << ","
             << job.model_fname << ","
             << job.model_id << ","
             << job.device_id << ","
             << job.start_idx << ","
             << job.end_idx << ","
             << job.subgraph_idx << ","
             << job.enqueue_time << ","
             << job.invoke_time << ","
             << job.end_time << ","
             << job.expected_execution_time_us << ","
             << job.actual_execution_time_us << ","
             << job.expected_latency_us << ","
             << job.actual_latency_us << ","
             << job.slo << ","
             << job.is_finished << ","
             << job. is_slo_violated << "\n";
  }
  log_file.close();
  lock.unlock();
}

void Planner::EnqueueFinishedJob(Job job) {
  std::unique_lock<std::mutex> lock(job_queue_mtx_);
  if (job.is_finished) {
    job.actual_latency_us = job.end_time - job.enqueue_time;
    job.actual_execution_time_us = job.end_time - job.invoke_time;
    job.is_slo_violated = (job.actual_latency_us > job.slo) ? true : false;
  }
  jobs_finished_.push_back(job);
  lock.unlock();

  end_invoke_.notify_one();
}

void Planner::EnqueueRequest(Job job) {
  job.enqueue_time = profiling::time::NowMicros();
  std::unique_lock<std::mutex> lock(requests_mtx_);
  requests_.push_back(job);
  num_submitted_jobs_++;
  lock.unlock();

  planner_safe_bool_.notify();
}

void Planner::EnqueueBatch(std::vector<Job> jobs) {
  std::unique_lock<std::mutex> lock(requests_mtx_);
  auto enqueue_time = profiling::time::NowMicros();
  for (Job job : jobs) {
    if (job.enqueue_time == 0) {
      job.enqueue_time = enqueue_time;
    }
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
