#include <fstream>

#include "tensorflow/lite/planner/planner.h"
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/tools/logging.h"
#include "tensorflow/lite/interpreter.h"

namespace tflite {
namespace impl {

Planner::Planner(Interpreter* interpreter) {
  interpreter_ = interpreter;
  planner_thread_ = std::thread([this]{this->Plan();});
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

void Planner::CopyToLocalQueue(JobQueue& local_jobs) {
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
}

void Planner::EnqueueToWorker(Job job) {
  Worker* worker = GetInterpreter()->GetWorker(job.device_id);
  if (worker == nullptr) return;
  {
    std::lock_guard<std::mutex> lock(worker->GetDeviceMtx());
    worker->GetDeviceRequests().push_back(job);
    worker->GetRequestCv().notify_one();
  }
}

void Planner::UpdateDeviceWaitingTime() {
  for (int i = 0; i < kTfLiteNumDevices; ++i) {
    TfLiteDeviceFlags device_flag = static_cast<TfLiteDeviceFlags>(i);
    Worker* worker = GetInterpreter()->GetWorker(device_flag);
    if (worker != nullptr) {
      device_waiting_[device_flag] = worker->GetWaitingTime();
    } else {
      device_waiting_[device_flag] = -1;
    }
  }
}

std::set<TfLiteDeviceFlags> Planner::GetIdleDevices() {
  std::set<TfLiteDeviceFlags> idle_devices;
  for (int i = 0; i < kTfLiteNumDevices; ++i) {
    TfLiteDeviceFlags device_flag = static_cast<TfLiteDeviceFlags>(i);
    if (device_waiting_[device_flag] == 0) {
      idle_devices.insert(device_flag);
    }
  }
  return idle_devices;
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
 
  if (job.is_final_subgraph) {
    jobs_finished_record_[GetJobRecordIndex(job.job_id)] = job;
    num_finished_jobs_++;
  }

  end_invoke_.notify_all();
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

      if (job.slo_us > 0 && job.is_final_subgraph &&
          job.status == kTfLiteJobSuccess) {
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

bool Planner::IsJobIdValid(int job_id) {
  return num_submitted_jobs_ - job_id <= NUM_FINISHED_RECORDS;
}

int Planner::GetJobRecordIndex(int job_id) const {
  return job_id % NUM_FINISHED_RECORDS;
}

void Planner::UpdateModelDeviceMapping() {
  std::set<int> models = GetInterpreter()->models();
  if (models.size() != model_device_map_.size()) {
    // (# of available devices, vector of model_id)
    std::map<int, std::set<int>> devices_per_models_map;
    for (auto model_id : models) {
      int count = 0;
      for (int device_idx = 0; device_idx < kTfLiteNumDevices; device_idx++) {
        if (GetInterpreter()->GetSubgraphIdx(
              model_id, static_cast<TfLiteDeviceFlags>(device_idx)) != -1) {
          count++;
        }
      }
      devices_per_models_map[count].insert(model_id);
    }

    int device_idx = 0;
    while (devices_per_models_map.size()) {
      // Loop through models in ascending order 
      // based on # of available devices
      // (Assign models that has limited support first)
      int selected_model_id = -1;
      for (auto& devices_per_models : devices_per_models_map) {
        for (int model_id : devices_per_models.second) {
          if (GetInterpreter()->GetSubgraphIdx(
                model_id, static_cast<TfLiteDeviceFlags>(device_idx)) != -1) {
            selected_model_id = model_id;
            break;
          }
        }

        if (selected_model_id != -1) {
          devices_per_models.second.erase(selected_model_id);
          if (devices_per_models.second.size() == 0)
            devices_per_models_map.erase(devices_per_models.first);
          break;
        }
      }

      if (selected_model_id != -1) {
        model_device_map_[selected_model_id] =
            static_cast<TfLiteDeviceFlags>(device_idx);
      }

      device_idx = (device_idx + 1) % kTfLiteNumDevices;
    };
  }
}


void Planner::Plan() {
  while (true) {
    if (GetSafeBool().wait()) {
       return;
    }
    UpdateModelDeviceMapping();
    for (size_t i = 0; i < GetNumQueues(); ++i) {
      UpdateDeviceWaitingTime();
      auto action = schedulers_[i].Schedule();
      EnqueueJobsToWorkers(action);
    }
  }
}

}  // namespace impl
}  // namespace tflite
