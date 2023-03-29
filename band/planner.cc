#include "band/planner.h"

#include <fstream>

#include "band/context.h"
#include "band/logger.h"
#include "band/model_spec.h"
#include "band/scheduler/fixed_worker_scheduler.h"
#include "band/scheduler/heterogeneous_earliest_finish_time_scheduler.h"
#include "band/scheduler/least_slack_first_scheduler.h"
#include "band/scheduler/round_robin_scheduler.h"
#include "band/scheduler/shortest_expected_latency_scheduler.h"
#include "band/time.h"
#include "planner.h"

namespace Band {
Planner::Planner(Context& context) : num_submitted_jobs_(0), context_(context) {
  planner_thread_ = std::thread([this] { this->Plan(); });
}

Planner::~Planner() {
  FlushFinishedJobs();
  planner_safe_bool_.terminate();
  planner_thread_.join();
}

absl::Status Planner::Init(const PlannerConfig& config) {
  schedule_window_size_ = config.schedule_window_size;
  log_path_ = config.log_path;
  if (log_path_.size()) {
    // Open file to write per-request timestamps later
    // NOTE: Columns starting `sched_id` are added for debugging purpose
    // and the metrics are only for ShortestExpectedLatency Planner.
    std::ofstream log_file(log_path_);
    if (!log_file.is_open()) {
      return absl::InternalError(absl::StrFormat(
          "[Planner] Failed to open log path %s", log_path_.c_str()));
    }
    log_file << "sched_id\t"
             << "job_id\t"
             << "model_name\t"
             << "model_id\t"
             << "worker_id\t"
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
  if (schedulers.size() == 0 || schedulers.size() > 2) {
    return absl::InternalError(absl::StrFormat(
        "[Planner] Not supported for %d schedulers", schedulers_.size()));
  }

  bool allow_fallback = false;
  local_queues_.resize(schedulers.size());
  for (int i = 0; i < schedulers.size(); ++i) {
    BAND_LOG_INTERNAL(BAND_LOG_INFO, "[Planner] create scheduler %d.",
                      schedulers[i]);
    if (schedulers[i] == SchedulerType::FixedWorker) {
      schedulers_.emplace_back(new FixedWorkerScheduler(context_));
    } else if (schedulers[i] == SchedulerType::FixedWorkerGlobalQueue) {
      schedulers_.emplace_back(new FixedWorkerGlobalQueueScheduler(context_));
    } else if (schedulers[i] == SchedulerType::RoundRobin) {
      schedulers_.emplace_back(new RoundRobinScheduler(context_));
    } else if (schedulers[i] == SchedulerType::ShortestExpectedLatency) {
      schedulers_.emplace_back(new ShortestExpectedLatencyScheduler(
          context_, schedule_window_size_));
    } else if (schedulers[i] ==
               SchedulerType::HeterogeneousEarliestFinishTime) {
      schedulers_.emplace_back(
          new HEFTScheduler(context_, schedule_window_size_, false));
    } else if (schedulers[i] == SchedulerType::LeastSlackTimeFirst) {
      schedulers_.emplace_back(
          new LeastSlackFirstScheduler(context_, schedule_window_size_));
    } else if (schedulers[i] ==
               SchedulerType::HeterogeneousEarliestFinishTimeReserved) {
      schedulers_.emplace_back(
          new HEFTScheduler(context_, schedule_window_size_, true));
    } else {
      return absl::InternalError("[Planner] Unsupported scheduler type.");
    }

    // Checks if all the schedulers have the same requirements for the
    // fallback subgraphs.
    // Currently, we do not allow using schedulers with different requirements
    // for the fallback subgraphs.
    if (i == 0) {
      allow_fallback = schedulers_[i]->NeedFallbackSubgraphs();
    } else if (allow_fallback != schedulers_[i]->NeedFallbackSubgraphs()) {
      return absl::InternalError(
          "[Planner] Different type of scheduler requirements.");
    }
  }

  // All schedulers must have the same worker type.
  if (GetWorkerType() == (static_cast<int>(WorkerType::DeviceQueue) |
                          static_cast<int>(WorkerType::GlobalQueue))) {
    return absl::InternalError(
        "All schedulers must have the same worker type.");
  }

  if (config.cpu_mask != CPUMaskFlags::All) {
    cpu_set_ = BandCPUMaskGetSet(config.cpu_mask);
    need_cpu_update_ = true;
  }

  return absl::OkStatus();
}

absl::Status Planner::AddScheduler(std::unique_ptr<IScheduler> scheduler) {
  schedulers_.emplace_back(std::move(scheduler));
  local_queues_.resize(schedulers_.size());
  return GetWorkerType() == (static_cast<int>(WorkerType::DeviceQueue) |
                             static_cast<int>(WorkerType::GlobalQueue))
             ? absl::InternalError(
                   "All schedulers must have the same worker type.")
             : absl::OkStatus();
}

JobId Planner::EnqueueRequest(Job job, bool push_front) {
  return EnqueueBatch({job}, push_front)[0];
}

std::vector<JobId> Planner::EnqueueBatch(std::vector<Job> jobs,
                                         bool push_front) {
  std::vector<JobId> job_ids(jobs.size());
  {
    std::unique_lock<std::mutex> request_lock(requests_.mtx);
    auto enqueue_time = Time::NowMicros();
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
    }

    auto insert_position =
        push_front ? requests_.queue.begin() : requests_.queue.end();
    requests_.queue.insert(insert_position, jobs.begin(), jobs.end());
  }
  planner_safe_bool_.notify();
  return job_ids;
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

void Planner::EnqueueFinishedJob(Job& job) {
  std::unique_lock<std::mutex> lock(jobs_finished_.mtx);
  jobs_finished_.queue.push_back(job);
  lock.unlock();

  std::lock_guard<std::mutex> request_lock(requests_.mtx);

  // record finished / failed job
  if (context_.IsEnd(job.subgraph_key) || job.status != JobStatus::Success) {
    jobs_finished_record_[GetJobRecordIndex(job.job_id)] = job;
    num_finished_jobs_++;

    end_invoke_.notify_all();
  }

  // report end invoke using callback
  if (on_end_request_ && job.require_callback &&
      context_.IsEnd(job.subgraph_key)) {
    on_end_request_(job.job_id, job.status == JobStatus::Success
                                    ? absl::OkStatus()
                                    : absl::InternalError("Job failed."));
  }
}

void Planner::PrepareReenqueue(Job& job) {
  job.invoke_time = 0;
  job.end_time = 0;
  job.resolved_unit_subgraphs = 0;
  job.following_jobs.clear();
}

bool Planner::NeedProfile() {
  for (int i = 0; i < schedulers_.size(); ++i) {
    if (schedulers_[i]->NeedProfile()) return true;
  }
  return false;
}

bool Planner::NeedFallbackSubgraphs() const {
  for (int i = 0; i < schedulers_.size(); ++i) {
    if (schedulers_[i]->NeedFallbackSubgraphs()) return true;
  }
  return false;
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

void Planner::SetOnEndRequest(
    std::function<void(int, absl::Status)> on_end_request) {
  on_end_request_ = on_end_request;
}

int Planner::GetWorkerType() const {
  int worker_type = 0;
  for (int i = 0; i < schedulers_.size(); ++i) {
    // TODO(widiba03304): Planner's worker type should not have an integer type.
    // Fix it to have a categorical type.
    worker_type |= static_cast<int>(schedulers_[i]->GetWorkerType());
  }
  return worker_type;
}

absl::Status Planner::Plan() {
  while (true) {
    if (planner_safe_bool_.wait()) {
      // TODO(widiba03304): Check correctness;
      continue;
    }

    if (need_cpu_update_) {
      if (!SetCPUThreadAffinity(cpu_set_).ok()) {
        return absl::InternalError("[Planner] Failed to set cpu thread affinity");
      }
      need_cpu_update_ = false;
    }
    CopyToLocalQueues();
    do {
      need_reschedule_ = false;
      for (size_t i = 0; i < local_queues_.size(); ++i) {
        schedulers_[i]->Schedule(local_queues_[i]);
      }
    } while (need_reschedule_);
  }
}

void Planner::FlushFinishedJobs() {
  std::lock_guard<std::mutex> queue_lock(jobs_finished_.mtx);
  std::ofstream log_file(log_path_, std::ofstream::app);
  if (log_file.is_open()) {
    while (!jobs_finished_.queue.empty()) {
      Job job = jobs_finished_.queue.front();
      jobs_finished_.queue.pop_front();

      bool is_final_subgraph = context_.IsEnd(job.subgraph_key);

      if (job.slo_us > 0 && is_final_subgraph &&
          job.status == JobStatus::Success) {
        // check if slo has been violated or not
        auto latency = job.end_time - job.enqueue_time;
        job.status =
            latency > job.slo_us ? JobStatus::SLOViolation : JobStatus::Success;
      }

      if (is_final_subgraph) {
        // update internal map to keep track of the # of inferences per model
        model_execution_count_[job.model_id]++;
      }

      std::string prev_subgraphs;

      // TODO: how to log these??
      // for (int i : job.previous_subgraph_indices) {
      //   prev_subgraphs += std::to_string(i) + " ";
      // }

      // write all timestamp statistics to log file
      log_file << job.sched_id << "\t" << job.job_id << "\t" << job.model_fname
               << "\t" << job.model_id << "\t" << job.subgraph_key.GetWorkerId()
               << "\t" << job.enqueue_time << "\t" << job.invoke_time << "\t"
               << job.end_time << "\t" << job.profiled_execution_time << "\t"
               << job.expected_execution_time << "\t" << job.slo_us << "\t"
               << job.status << "\t" << is_final_subgraph << "\t"
               << prev_subgraphs << "\n";
    }
    log_file.close();
  } else {
    BAND_REPORT_ERROR(context_.GetErrorReporter(),
                      "[Planner] Invalid log file path %s", log_path_.c_str());
  }
}

void Planner::CopyToLocalQueues() {
  std::unique_lock<std::mutex> request_lock(GetRequestsMtx());
  JobQueue& requests = GetRequests();
  if (!requests.empty()) {
    if (schedulers_.size() == 1) {
      // Gets jobs from requests and removes those jobs from the requests.
      auto& local_jobs = local_queues_[0];
      local_jobs.insert(local_jobs.end(),
                        std::make_move_iterator(requests.begin()),
                        std::make_move_iterator(requests.end()));
    } else if (schedulers_.size() == 2) {
      // TODO: general method for assigning SLO/non-SLO requests
      for (Job& job : requests) {
        if (job.slo_us > 0) {
          local_queues_[0].push_back(std::move(job));
        } else {
          local_queues_[1].push_back(std::move(job));
        }
      }
    }  // other else cases should have been caught in Init()

    requests.clear();
  }
  request_lock.unlock();
}

void Planner::EnqueueToWorker(const std::vector<ScheduleAction>& actions) {
  for (auto& action : actions) {
    Job job;
    SubgraphKey target_key;

    std::tie(job, target_key) = action;

    Worker* worker = context_.GetWorker(target_key.GetWorkerId());
    if (worker == nullptr) {
      BAND_REPORT_ERROR(
          context_.GetErrorReporter(),
          "EnqueueToWorker failed. Requests scheduled to null worker id %d",
          target_key.GetWorkerId());
      return;
    }

    if (IsSLOViolated(job)) {
      HandleSLOViolatedJob(job);
      continue;
    }

    {
      std::lock_guard<std::mutex> lock(worker->GetDeviceMtx());

      if (worker->IsEnqueueReady()) {
        UpdateJobScheduleStatus(job, target_key);
        worker->EnqueueJob(job);
      } else {
        EnqueueRequest(job, true);
      }
    }

    worker->GetRequestCv().notify_one();
  }
}

bool Planner::IsSLOViolated(Job& job) {
  if (job.status == JobStatus::SLOViolation) {
    return true;
  }
  // this job has an SLO; check if it's not too late already
  if (job.slo_us > 0) {
    WorkerWaitingTime workers_waiting = context_.GetWorkerWaitingTime();
    int64_t current_time = Time::NowMicros();
    int64_t expected_latency = workers_waiting[job.subgraph_key.GetWorkerId()] +
                               job.expected_execution_time;

    if (current_time + expected_latency > job.enqueue_time + job.slo_us) {
      return true;
    }
  }
  return false;
}

void Planner::HandleSLOViolatedJob(Job& job) {
  // no point in running this job anymore
  job.status = JobStatus::SLOViolation;

  // mark this as -1 to differentiate it from the default value, 0
  job.invoke_time = -1;

  // mark the time of this decision (of early-dropping this job)
  job.end_time = Time::NowMicros();
  EnqueueFinishedJob(job);

  // Set reschedule flag.
  need_reschedule_ = true;
}

void Planner::UpdateJobScheduleStatus(Job& job, const SubgraphKey& target_key) {
  job.subgraph_key = target_key;
  job.sched_id = IssueSchedId();
  job.profiled_execution_time = context_.GetProfiled(target_key);
  job.expected_execution_time = context_.GetExpected(target_key);
  job.resolved_unit_subgraphs |= target_key.GetUnitIndices();

  if (!context_.IsEnd(target_key)) {
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
    remaining_ops.resolved_unit_subgraphs = job.resolved_unit_subgraphs;
    remaining_ops.previous_subgraph_keys = job.previous_subgraph_keys;
    remaining_ops.previous_subgraph_keys.emplace_back(job.subgraph_key);

    job.following_jobs.clear();
    job.following_jobs.push_back(remaining_ops);
  }
}

bool Planner::IsJobIdValid(int job_id) {
  return num_submitted_jobs_ - job_id <= NUM_FINISHED_RECORDS;
}

int Planner::GetJobRecordIndex(int job_id) const {
  return job_id % NUM_FINISHED_RECORDS;
}

}  // namespace Band
