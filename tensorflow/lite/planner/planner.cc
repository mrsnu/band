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
  log_file << "sched_id\t"
           << "model_name\t"
           << "model_id\t"
           << "device_id\t"
           << "start_idx\t"
           << "end_idx\t"
           << "subgraph_idx\t"
           << "enqueue_time\t"
           << "invoke_time\t"
           << "end_time\t"
           << "expected_exec_time\t"
           << "expected_latency\t"
           << "slo\t"
           << "is_final_subgraph\n";
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
    log_file << job.sched_id << "\t"
             << job.model_fname << "\t"
             << job.model_id << "\t"
             << job.device_id << "\t"
             << job.start_idx << "\t"
             << job.end_idx << "\t"
             << job.subgraph_idx << "\t"
             << job.enqueue_time << "\t"
             << job.invoke_time << "\t"
             << job.end_time << "\t"
             << job.expected_exec_time << "\t"
             << job.expected_latency << "\t"
             << job.slo << "\t"
             << job.is_final_subgraph << "\n";
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
  if (job.enqueue_time == 0) {
    // job.enqueue_time may already be set if this model contains a fallback
    // op, in which case we do not overwrite the set value
    job.enqueue_time = profiling::time::NowMicros();
  }
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
      // job.enqueue_time may already be set if this model contains a fallback
      // op, in which case we do not overwrite the set value
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
