#include "tensorflow/lite/planner/planner.h"

#include <fstream>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/planner/fixed_device_scheduler.h"
#include "tensorflow/lite/planner/round_robin_scheduler.h"
#include "tensorflow/lite/planner/shortest_expected_latency_scheduler.h"
#include "tensorflow/lite/planner/heterogeneous_earliest_finish_time_scheduler.h"
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace impl {

Planner::Planner(Interpreter* interpreter) : num_submitted_jobs_(0) {
  interpreter_ = interpreter;
  planner_thread_ = std::thread([this] { this->Plan(); });
}

Planner::~Planner() {
  FlushFinishedJobs();
  planner_safe_bool_.terminate();
  planner_thread_.join();
}

TfLiteStatus Planner::Init(PlannerConfig& config) {
  schedule_window_size_ = config.schedule_window_size;
  log_path_ = config.log_path;
  if (log_path_.size()) {
    // Open file to write per-request timestamps later
    // NOTE: Columns starting `sched_id` are added for debugging purpose
    // and the metrics are only for ShortestExpectedLatency Planner.
    std::ofstream log_file(log_path_);
    if (!log_file.is_open()) return kTfLiteError;
    log_file << "sched_id\t"
             << "job_id\t"
             << "model_name\t"
             << "model_id\t"
             << "device_id\t"
             << "worker_id\t"
             << "subgraph_idx\t"
             << "enqueue_time\t"
             << "invoke_time\t"
             << "end_time\t"
             << "profiled_execution_time\t"
             << "expected_execution_time\t"
             << "slo_us\t"
             << "job_status\t"
             << "is_final_subgraph\t"
             << "prev_subgraphs\n";
    log_file.close();
  }

  auto& schedulers = config.schedulers;
  local_queues_.resize(schedulers.size());
  bool allow_fallback;
  for (int i = 0; i < schedulers.size(); ++i) {
    if (schedulers[i] == kFixedDevice) {
      schedulers_.emplace_back(new FixedDeviceScheduler(this));
    } else if (schedulers[i] == kFixedDeviceGlobalQueue) {
      schedulers_.emplace_back(new FixedDeviceGlobalQueueScheduler(this));
    } else if (schedulers[i] == kRoundRobin) {
      schedulers_.emplace_back(new RoundRobinScheduler(this));
    } else if (schedulers[i] == kShortestExpectedLatency) {
      schedulers_.emplace_back(new ShortestExpectedLatencyScheduler(this));
    } else if (schedulers[i] == kHeterogeneousEarliestFinishTime) {
      schedulers_.emplace_back(new HeterogeneousEarliestFinishTimeScheduler(this));
    } else {
      return kTfLiteError;
    }

    // Checks if all the schedulers have the same requirements for the
    // fallback subgraphs.
    // Currently, we do not allow using schedulers with different requirements
    // for the fallback subgraphs.
    if (i == 0) {
      allow_fallback = schedulers_[i]->NeedFallbackSubgraphs();
    } else if (allow_fallback != schedulers_[i]->NeedFallbackSubgraphs()) {
      return kTfLiteError;
    }
  }

  // All schedulers must have the same worker type.
  if (GetWorkerType() == (kDeviceQueue | kGlobalQueue)) {
    return kTfLiteError;
  }

  if (config.cpu_masks != impl::kTfLiteAll) {
    cpu_set_ = impl::TfLiteCPUMaskGetSet(config.cpu_masks);
    need_cpu_update_ = true;
  }

  return kTfLiteOk;
}

bool Planner::NeedProfile() {
  for (int i = 0; i < schedulers_.size(); ++i) {
    if (schedulers_[i]->NeedProfile()) return true;
  }
  return false;
}

int Planner::GetWorkerType() const {
  int worker_type = 0;
  for (int i = 0; i < schedulers_.size(); ++i) {
    worker_type |= schedulers_[i]->GetWorkerType();
  }
  return worker_type;
}

bool Planner::NeedFallbackSubgraphs() const {
  for (int i = 0; i < schedulers_.size(); ++i) {
    if (schedulers_[i]->NeedFallbackSubgraphs()) return true;
  }
  return false;
}

void Planner::CopyToLocalQueue(JobQueue& local_jobs) {
  std::unique_lock<std::mutex> request_lock(GetRequestsMtx());
  JobQueue& requests = GetRequests();
  if (!requests.empty()) {
    // Gets jobs from requests and removes those jobs from the requests.
    local_jobs.insert(local_jobs.end(),
                      std::make_move_iterator(requests.begin()),
                      std::make_move_iterator(requests.end()));
    requests.clear();
  }
  request_lock.unlock();
}

bool Planner::IsSLOViolated(Job& job) {
  // this job has an SLO; check if it's not too late already
  if (job.slo_us > 0) {
    int64_t current_time = profiling::time::NowMicros();
    int64_t expected_latency =
        workers_waiting_[job.worker_id] +
        job.profiled_execution_time;

    if (current_time + expected_latency > job.enqueue_time + job.slo_us) {
      return true;
    }
  }
  return false;
}

void Planner::HandleSLOViolatedJob(Job& job) {
  // no point in running this job anymore
  job.status = kTfLiteJobSLOViolation;

  // mark this as -1 to differentiate it from the default value, 0
  job.invoke_time = -1;

  // mark the time of this decision (of early-dropping this job)
  job.end_time = profiling::time::NowMicros();
  EnqueueFinishedJob(job);

  // Set reschedule flag.
  need_reschedule_ = true;
}

void Planner::EnqueueToWorkers(ScheduleAction& action) {
  for (auto& queue : action) {
    auto device = queue.first;
    auto& requests = queue.second;

    if (requests.empty()) {
      continue;
    }

    Worker* worker = GetInterpreter()->GetWorker(device);
    if (worker == nullptr) return;
    {
      std::lock_guard<std::mutex> lock(worker->GetDeviceMtx());
      for (auto request : requests) {
        if (IsSLOViolated(request)) {
          HandleSLOViolatedJob(request);
          continue;
        }
        if (!worker->GiveJob(request)) {
          PrepareReenqueue(request);
          EnqueueRequest(request, true);
        }
      }
      worker->GetRequestCv().notify_one();
    }
    requests.clear();
  }
}

void Planner::UpdateWorkerWaitingTime() {
  for (int i = 0; i < GetInterpreter()->GetNumWorkers(); ++i) {
    workers_waiting_[i] = GetInterpreter()->GetWorker(i)->GetWaitingTime();
  }
}

std::set<int> Planner::GetIdleWorkers() {
  std::set<int> idle_workers;
  for (int i = 0; i < GetInterpreter()->GetNumWorkers(); ++i) {
    if (!GetInterpreter()->GetWorker(i)->IsBusy()) {
      idle_workers.insert(i);
    }
  }
  return idle_workers;
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

  // record finished / failed job
  if (interpreter_->subgraph(job.subgraph_idx)->IsEnd() ||
      job.status != kTfLiteJobSuccess) {
    jobs_finished_record_[GetJobRecordIndex(job.job_id)] = job;
    num_finished_jobs_++;

    end_invoke_.notify_all();
  }
}

int Planner::EnqueueRequest(Job job, bool push_front) {
  return EnqueueBatch({job}, push_front)[0];
}

std::vector<int> Planner::EnqueueBatch(std::vector<Job> jobs, bool push_front) {
  std::vector<int> job_ids(jobs.size());
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
  }

  std::unique_lock<std::mutex> request_lock(requests_.mtx);
  auto insert_position =
      push_front ? requests_.queue.begin() : requests_.queue.end();
  requests_.queue.insert(insert_position, jobs.begin(), jobs.end());
  request_lock.unlock();

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
          interpreter_->subgraph(job.subgraph_idx)->IsEnd();

      if (job.slo_us > 0 && is_final_subgraph &&
          job.status == kTfLiteJobSuccess) {
        // check if slo has been violated or not
        auto latency = job.end_time - job.enqueue_time;
        job.status =
            latency > job.slo_us ? kTfLiteJobSLOViolation : kTfLiteJobSuccess;
      }

      if (is_final_subgraph) {
        // update internal map to keep track of the # of inferences per model
        model_execution_count_[job.model_id]++;
      }

      std::string prev_subgraphs;

      for (int i : job.previous_subgraph_indices) {
        prev_subgraphs += std::to_string(i) + " ";
      }

      // write all timestamp statistics to log file
      log_file << job.sched_id << "\t"
               << job.job_id << "\t"
               << job.model_fname << "\t"
               << job.model_id << "\t"
               << job.device_id << "\t"
               << job.worker_id << "\t"
               << job.subgraph_idx << "\t"
               << job.enqueue_time << "\t"
               << job.invoke_time << "\t"
               << job.end_time << "\t"
               << job.profiled_execution_time << "\t"
               << job.expected_execution_time << "\t"
               << job.slo_us << "\t"
               << job.status << "\t"
               << is_final_subgraph << "\t"
               << prev_subgraphs << "\n";
    }
    log_file.close();
  } else {
    TFLITE_LOG(ERROR) << "Invalid log file path :" << log_path_;
  }
}

int Planner::IssueSchedId() { return sched_id_++; }

void Planner::UpdateJobScheduleStatus(Job& job, Subgraph* target_subgraph) {
  SubgraphKey& target_key = target_subgraph->GetKey();
  job.subgraph_idx = interpreter_->GetSubgraphIdx(target_key);
  job.worker_id = target_key.worker_id;
  job.device_id = interpreter_->GetWorkerDeviceFlag(target_key.worker_id);
  job.sched_id = IssueSchedId();
  job.profiled_execution_time = interpreter_->GetProfiledLatency(target_key);
  job.expected_execution_time = interpreter_->GetExpectedLatency(job.subgraph_idx);

  if (!target_subgraph->IsEnd()) {
    Job remaining_ops(job.model_id);
    remaining_ops.model_fname = job.model_fname;
    remaining_ops.slo_us = job.slo_us;
    remaining_ops.enqueue_time = job.enqueue_time;
    remaining_ops.following_jobs = job.following_jobs;
    remaining_ops.expected_latency = job.expected_latency;
    remaining_ops.sched_id = job.sched_id;
    remaining_ops.job_id = job.job_id;
    remaining_ops.input_handle = job.input_handle;
    remaining_ops.output_handle = job.output_handle;
    remaining_ops.resolved_tensors = job.resolved_tensors;
    remaining_ops.previous_subgraph_indices = job.previous_subgraph_indices;
    remaining_ops.previous_subgraph_indices.emplace_back(job.subgraph_idx);
    // The next start_unit_idx is the next one from the current max unit index.
    remaining_ops.start_unit_idx = *target_key.unit_indices.rbegin() + 1;

    for (int output_index : target_subgraph->outputs()) {
      remaining_ops.resolved_tensors.insert(output_index);
    }

    job.following_jobs.clear();
    job.following_jobs.push_back(remaining_ops);
  }
}

void Planner::PrepareReenqueue(Job& job) {
  job.invoke_time = 0;
  job.end_time = 0;
  job.following_jobs.clear();
}

bool Planner::IsJobIdValid(int job_id) {
  return num_submitted_jobs_ - job_id <= NUM_FINISHED_RECORDS;
}

int Planner::GetJobRecordIndex(int job_id) const {
  return job_id % NUM_FINISHED_RECORDS;
}

void Planner::TryUpdateModelWorkerMapping() {
  std::set<int> models = GetInterpreter()->models();
  const int num_workers = GetInterpreter()->GetNumWorkers();
  if (models.size() > model_worker_map_.size()) {
    // (# of available workers, vector of model_id)
    std::map<int, std::set<int>> workers_per_models_map;
    for (auto model_id : models) {
      int count = 0;
      for (int worker_id = 0; worker_id < num_workers; ++worker_id) {
        if (GetInterpreter()->GetSubgraphIdx(
                model_id, worker_id) != -1) {
          count++;
        }
      }
      workers_per_models_map[count].insert(model_id);
    }

    int worker_id = 0;
    while (workers_per_models_map.size()) {
      // Loop through models in ascending order
      // based on # of available devices
      // (Assign models that has limited support first)
      int selected_model_id = -1;
      for (auto& workers_per_models : workers_per_models_map) {
        for (int model_id : workers_per_models.second) {
          if (GetInterpreter()->GetSubgraphIdx(
                  model_id, worker_id) != -1) {
            selected_model_id = model_id;
            break;
          }
        }

        if (selected_model_id != -1) {
          workers_per_models.second.erase(selected_model_id);
          if (workers_per_models.second.size() == 0)
            workers_per_models_map.erase(workers_per_models.first);
          break;
        }
      }

      if (selected_model_id != -1) {
        model_worker_map_[selected_model_id] = worker_id;
      }

      worker_id = (worker_id + 1) % num_workers;
    };
  }
}

void Planner::Plan() {
  while (true) {
    if (GetSafeBool().wait()) {
      return;
    }

    if (need_cpu_update_) {
      if (SetCPUThreadAffinity(cpu_set_) != kTfLiteOk) {
        TFLITE_LOG(ERROR) << "Planner failed to set cpu thread affinity";
        // TODO #21: Handle errors in multi-thread environment
      } 
      need_cpu_update_ = false;
    }

    CopyToLocalQueue(local_queues_[0]);
    TryUpdateModelWorkerMapping();
    do {
      need_reschedule_ = false;
      for (size_t i = 0; i < local_queues_.size(); ++i) {
        schedulers_[i]->Schedule(local_queues_[i]);
      }
    } while(need_reschedule_);
  }
}

void Scheduler::EnqueueAction(Job job, Subgraph* subgraph) {
  planner_->UpdateJobScheduleStatus(job, subgraph);
  action_[subgraph->GetKey().worker_id].push_back(job);
  planner_->EnqueueToWorkers(action_);
}

}  // namespace impl
}  // namespace tflite
