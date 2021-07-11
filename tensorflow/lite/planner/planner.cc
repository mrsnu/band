#include <fstream>

#include "tensorflow/lite/planner/planner.h"
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/tools/logging.h"
#include "tensorflow/lite/interpreter.h"

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

JobQueue Planner::CopyToLocalQueue() {
  JobQueue local_jobs;
  std::unique_lock<std::mutex> request_lock(GetRequestsMtx());
  JobQueue& requests = GetRequests();
  if (!requests.empty()) {
    // Gets the specific amount of jobs from requests
    // and removes those jobs from the requests.
    int window_size = std::min(GetWindowSize(), (int) requests.size());
    local_jobs.insert(local_jobs.begin(), std::make_move_iterator(requests.begin()), std::make_move_iterator(requests.begin() + window_size));
    requests.erase(requests.begin(), requests.begin() + window_size);
  }
  request_lock.unlock();

  return local_jobs;
}

void Planner::Wait(std::vector<int> job_ids) {
  if (job_ids.size() == 0) {
    return;
  }

  std::unique_lock<std::mutex> request_lock(requests_.mtx);
  end_invoke_.wait(request_lock, [this, job_ids] {
    for (int job_id : job_ids) {
      if (!IsJobIdValid(job_id)) {
        continue;
      }
      if (jobs_finished_record_[GetJobRecordIndex(job_id)].job_id != job_id) {
        return false;
      }
    }
    return true;
  });

  request_lock.unlock();
  FlushFinishedJobs();
}

void Planner::WaitAll() {
  std::unique_lock<std::mutex> request_lock(requests_.mtx);
  end_invoke_.wait(request_lock, [this]() {
    return num_finished_jobs_ >= num_submitted_jobs_;
  });

  request_lock.unlock();

  FlushFinishedJobs();
}

void Planner::EnqueueFinishedJob(Job job) {
  std::unique_lock<std::mutex> lock(jobs_finished_.mtx);
  jobs_finished_.queue.push_back(job);
  lock.unlock();

  std::lock_guard<std::mutex> request_lock(requests_.mtx);
  
  if (!interpreter_->subgraph(job.subgraph_idx)->GetNextSubgraph()) {
    jobs_finished_record_[GetJobRecordIndex(job.job_id)] = job;
    num_finished_jobs_++;

    end_invoke_.notify_all();
  }
}

int Planner::EnqueueRequest(Job job) { return EnqueueBatch({job})[0]; }

std::vector<int> Planner::EnqueueBatch(std::vector<Job> jobs) {
  std::vector<int> job_ids(jobs.size());
  std::unique_lock<std::mutex> lock(requests_.mtx);
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
      job.resolved_tensors =
          interpreter_->GetModelSpec(job.model_id).input_tensors;
    }
    job_ids[i] = job.job_id;
    requests_.queue.push_back(job);
  }
  lock.unlock();

  planner_safe_bool_.notify();

  return job_ids;
}

void Planner::SetWindowSize(int schedule_window_size) {
  schedule_window_size_ = schedule_window_size;
}

Job Planner::GetFinishedJob(int job_id) {
  std::lock_guard<std::mutex> request_lock(requests_.mtx);
  if (IsJobIdValid(job_id) &&
      jobs_finished_record_[GetJobRecordIndex(job_id)].job_id != -1) {
    return jobs_finished_record_[GetJobRecordIndex(job_id)];
  } else {
    return Job();
  }
}

void Planner::FlushFinishedJobs() {
  std::lock_guard<std::mutex> queue_lock(jobs_finished_.mtx);
  std::ofstream log_file(log_path_, std::ofstream::app);
  if (log_file.is_open()) {
    while (!jobs_finished_.queue.empty()) {
      Job job = jobs_finished_.queue.front();
      jobs_finished_.queue.pop_front();

      bool is_final_subgraph =
          interpreter_->subgraph(job.subgraph_idx)->GetNextSubgraph() == nullptr;

      if (job.slo_us > 0 && is_final_subgraph && job.status == kTfLiteJobSuccess) {
        // check if slo has been violated or not
        auto latency = job.end_time - job.enqueue_time;
        job.status =
            latency > job.slo_us ? kTfLiteJobSLOViolation : kTfLiteJobSuccess;
      }

      if (is_final_subgraph) {
        // update internal map to keep track of the # of inferences per model
        model_execution_count_[job.model_id]++;
      }

      // write all timestamp statistics to log file
      log_file << job.sched_id << "\t"
              << job.model_fname << "\t"
              << job.model_id << "\t"
              << job.device_id << "\t"
              << job.subgraph_idx << "\t"
              << job.enqueue_time << "\t"
              << job.invoke_time << "\t"
              << job.end_time << "\t"
              << job.profiled_time << "\t"
              << job.expected_latency << "\t"
              << job.slo_us << "\t"
              << job.status << "\t"
              << is_final_subgraph << "\n";
    }
    log_file.close();
  } else {
    TFLITE_LOG(ERROR) << "Invalid log file path :" << log_path_;
  }
}

bool Planner::IsJobIdValid(int job_id) {
  return num_submitted_jobs_ - job_id <= NUM_FINISHED_RECORDS;
}

int Planner::GetJobRecordIndex(int job_id) const {
  return job_id % NUM_FINISHED_RECORDS;
}

}  // namespace impl
}  // namespace tflite
