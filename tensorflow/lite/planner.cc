#include "tensorflow/lite/planner.h"
#include "tensorflow/lite/profiling/time.h"
#include <iostream>

namespace tflite {
namespace impl {

Planner::Planner(Interpreter* interpreter) {
  interpreter_ = interpreter;
  log_file_.open(log_path_, std::fstream::app);
  log_file_ << "batch_id\tjob_id\tmodel_name\tmodel_id\tdevice_id\tenqueue_time\tinvoke_time\tend_time\t";
  log_file_ << "cpu_waiting\tcpu_latency\tgpu_waiting\tgpu_latency\tdsp_waiting\tdsp_latency\tnpu_waiting\tnpu_latency\n";
  log_file_.close();
  wait_thread_ = std::thread([this]{this->Wait();});
}

Planner::~Planner() {
  planner_safe_bool_.terminate();
  planner_thread_.join();

  wait_safe_bool_.terminate();
  wait_thread_.join();
}


void Planner::Wait() {
  std::ofstream log_file(log_path_, std::ofstream::app);

  while (true) {
    if (wait_safe_bool_.wait())
      return;

    std::deque<Job> local_jobs;
    std::unique_lock<std::mutex> lock(job_queue_mtx_);
    if (jobs_finished_.empty()) {
      continue;
    } else {
      jobs_finished_.swap(local_jobs);
    }

    lock.unlock();

    std::unique_lock<std::mutex> cnt_lock(cnt_mtx_);
    for (Job& job : local_jobs) {
      cnt_++;
      auto it = callbacks_.find(job.job_id_);
      if (it != callbacks_.end()) {
        it->second();
        callbacks_.erase(it);
      }

      // write all timestamp statistics to log file
      log_file << job.batch_id_ << "\t"
               << job.job_id_ << "\t"
               << job.model_fname << "\t"
               << job.model_id_ << "\t"
               << job.device_id_ << "\t"
               << job.enqueue_time_ << "\t"
               << job.invoke_time_ << "\t"
               << job.end_time_ << "\t";

      for (int i = 0; i < 4; ++i) {
        log_file << job.waiting_time[i] << "\t";
        log_file << job.expected_latency[i];

        if (i == 3) {
          log_file << "\n";
        } else {
          log_file << "\t";
        }
      }
    }
    cnt_lock.unlock();
  }

  log_file.close();
}

void Planner::EnqueueFinishedJob(Job job) {
  std::unique_lock<std::mutex> lock(job_queue_mtx_);
  jobs_finished_.push_back(job);
  lock.unlock();

  wait_safe_bool_.notify();
}

void Planner::EnqueueRequest(Job job, std::function<void()> callback) {
  job.enqueue_time_ = profiling::time::NowMicros();
  std::unique_lock<std::mutex> lock(requests_mtx_);
  job.job_id_ = global_job_id_++;
  if (callback) {
    callbacks_[job.job_id_] = callback;
  }
  requests_.push_back(job);
  lock.unlock();

  planner_safe_bool_.notify();
}

void Planner::EnqueueBatch(std::vector<Job> jobs, std::vector<std::function<void()>> callbacks) {
  std::unique_lock<std::mutex> lock(requests_mtx_);
  auto enqueue_time = profiling::time::NowMicros();
  for (Job job : jobs) {
    job.job_id_ = global_job_id_++;
    job.enqueue_time_ = enqueue_time;
    requests_.push_back(job);
  }
  lock.unlock();

  planner_safe_bool_.notify();
}

}  // namespace impl
}  // namespace tflite
