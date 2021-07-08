#include "tensorflow/lite/planner/planner.h"
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/tools/logging.h"
#include "tensorflow/lite/interpreter.h"
#include <fstream>

namespace tflite {
namespace impl {

Planner::Planner(Interpreter* interpreter) {
  interpreter_ = interpreter;
}

Planner::~Planner() {
  FlushFinishedJobs();
  planner_safe_bool_.terminate();
  planner_thread_.join();
}

TfLiteStatus Planner::Init(PlannerConfig& config) {
  schedule_window_size_ = config.schedule_window_size;
  log_path_ = config.log_path;

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
           << "profiled_time\t"
           << "expected_latency\t"
           << "slo_us\t"
           << "job status\t"
           << "is_final_subgraph\n";
  log_file.close();
  
  return kTfLiteOk;
}

void Planner::Wait() {
  std::unique_lock<std::mutex> lock(job_queue_mtx_);
  end_invoke_.wait(lock, [this]{
    return jobs_finished_.size() >= num_submitted_jobs_;
  });

  FlushFinishedJobs();
}

void Planner::WaitAll() {
  std::unique_lock<std::mutex> record_lock(record_mtx_);
  end_invoke_.wait(record_lock, [this]() {
    return jobs_finished_record_.size() >= num_submitted_jobs_;
  });

  FlushFinishedJobs();
}

void Planner::FlushFinishedJobs() {
  std::lock_guard<std::mutex> queue_lock(job_queue_mtx_);
  std::ofstream log_file(log_path_, std::ofstream::app);
  if (log_file.is_open()) {
    while (!jobs_finished_.empty()) {
    Job job = jobs_finished_.front();
    jobs_finished_.pop_front();

    if (job.slo_us > 0 && job.is_final_subgraph && job.status == kTfLiteJobSuccess) {
      // check if slo has been violated or not
      auto latency = job.end_time - job.enqueue_time;
      job.status =
          latency > job.slo_us ? kTfLiteJobSLOViolation : kTfLiteJobSuccess;
    }

    if (job.end_idx == interpreter_->GetModelSpec(job.model_id).num_ops - 1) {
      // update internal map to keep track of the # of inferences per model
      model_execution_count_[job.model_id]++;
    }

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
             << job.profiled_time << "\t"
             << job.expected_latency << "\t"
             << job.slo_us << "\t"
             << job.status << "\t"
             << job.is_final_subgraph << "\n";
    }
    log_file.close();
  } else {
    TFLITE_LOG(ERROR) << "Invalid log file path :" << log_path_;
  }
}

void Planner::EnqueueFinishedJob(Job job) {
  std::unique_lock<std::mutex> lock(job_queue_mtx_);
  jobs_finished_.push_back(job);
  lock.unlock();

  end_invoke_.notify_one();
}

int Planner::EnqueueRequest(Job job) { return EnqueueBatch({job})[0]; }

std::vector<int> Planner::EnqueueBatch(std::vector<Job> jobs) {
  std::vector<int> job_ids(jobs.size());
  std::unique_lock<std::mutex> lock(requests_mtx_);
  auto enqueue_time = profiling::time::NowMicros();
  for (int i = 0; i < jobs.size(); i++) {
    Job& job = jobs[i];
    if (job.enqueue_time == 0) {
      // job.enqueue_time may already be set if this model contains a fallback
      // op, in which case we do not overwrite the set value
      job.enqueue_time = enqueue_time;
    }
    if (job.job_id == -1) {
      job.job_id = num_submitted_jobs_++;
    }
    job_ids[i] = job.job_id;
    requests_.push_back(job);
  }
  lock.unlock();

  planner_safe_bool_.notify();

  return job_ids;
}

}  // namespace impl
}  // namespace tflite
