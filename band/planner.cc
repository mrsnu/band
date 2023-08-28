#include "band/planner.h"

#include <fstream>

#include "absl/strings/str_format.h"
#include "band/engine_interface.h"
#include "band/job_tracer.h"
#include "band/logger.h"
#include "band/model_spec.h"
#include "band/scheduler/fixed_worker_scheduler.h"
#include "band/scheduler/heterogeneous_earliest_finish_time_scheduler.h"
#include "band/scheduler/least_slack_first_scheduler.h"
#include "band/scheduler/round_robin_idle_scheduler.h"
#include "band/scheduler/round_robin_scheduler.h"
#include "band/scheduler/shortest_expected_latency_scheduler.h"
#include "band/scheduler/thermal_scheduler.h"
#include "band/time.h"


namespace band {

Planner::Planner(IEngine& engine) : num_submitted_jobs_(0), engine_(engine) {
  planner_thread_ = std::thread([this] {
    device::Root();
    auto status = this->Plan();
    if (!status.ok()) {
      BAND_LOG_PROD(BAND_LOG_ERROR, "Planner thread failed: %s",
                    status.message());
    }
  });
}

Planner::~Planner() {
  if (log_path_.size()) {
    BAND_LOG_PROD(BAND_LOG_INFO, "Dumping job tracer to %s", log_path_.c_str());
    BAND_TRACER_DUMP(log_path_);
  }
  planner_safe_bool_.terminate();
  planner_thread_.join();
}

absl::Status Planner::Init(const PlannerConfig& config) {
  schedule_window_size_ = config.schedule_window_size;
  log_path_ = config.log_path;
  BAND_LOG_PROD(BAND_LOG_INFO, "Planner log path: %s", log_path_.c_str());

  auto& schedulers = config.schedulers;
  if (schedulers.size() == 0 || schedulers.size() > 2) {
    return absl::InternalError(absl::StrFormat(
        "[Planner] Not supported for %d schedulers", schedulers_.size()));
  }

  bool allow_fallback = false;
  for (int i = 0; i < schedulers.size(); ++i) {
    BAND_LOG_INTERNAL(BAND_LOG_INFO, "[Planner] create scheduler %d.",
                      schedulers[i]);
    if (schedulers[i] == SchedulerType::kFixedWorker) {
      schedulers_.emplace_back(new FixedWorkerScheduler(engine_));
    } else if (schedulers[i] == SchedulerType::kFixedWorkerGlobalQueue) {
      schedulers_.emplace_back(new FixedWorkerGlobalQueueScheduler(engine_));
    } else if (schedulers[i] == SchedulerType::kRoundRobin) {
      schedulers_.emplace_back(new RoundRobinScheduler(engine_));
    } else if (schedulers[i] == SchedulerType::kRoundRobinIdle) {
      schedulers_.emplace_back(new RoundRobinIdleScheduler(engine_));
    } else if (schedulers[i] == SchedulerType::kShortestExpectedLatency) {
      schedulers_.emplace_back(
          new ShortestExpectedLatencyScheduler(engine_, schedule_window_size_));
    } else if (schedulers[i] ==
               SchedulerType::kHeterogeneousEarliestFinishTime) {
      schedulers_.emplace_back(
          new HEFTScheduler(engine_, schedule_window_size_, false));
    } else if (schedulers[i] == SchedulerType::kLeastSlackTimeFirst) {
      schedulers_.emplace_back(
          new LeastSlackFirstScheduler(engine_, schedule_window_size_));
    } else if (schedulers[i] ==
               SchedulerType::kHeterogeneousEarliestFinishTimeReserved) {
      schedulers_.emplace_back(
          new HEFTScheduler(engine_, schedule_window_size_, true));
    } else if (schedulers[i] == SchedulerType::kThermal) {
      schedulers_.emplace_back(new ThermalScheduler(engine_));
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
  if (GetWorkerType() == (static_cast<int>(WorkerType::kDeviceQueue) |
                          static_cast<int>(WorkerType::kGlobalQueue))) {
    return absl::InternalError(
        "All schedulers must have the same worker type.");
  }

  if (config.cpu_mask != CPUMaskFlag::kAll) {
    cpu_set_ = BandCPUMaskGetSet(config.cpu_mask);
    need_cpu_update_ = true;
  }

  return absl::OkStatus();
}

absl::Status Planner::AddScheduler(std::unique_ptr<IScheduler> scheduler) {
  schedulers_.emplace_back(std::move(scheduler));
  return GetWorkerType() == (static_cast<int>(WorkerType::kDeviceQueue) |
                             static_cast<int>(WorkerType::kGlobalQueue))
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
    auto enqueue_time = time::NowMicros();
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

  std::unique_lock<std::mutex> finished_lock(job_finished_mtx_);
  end_invoke_.wait(finished_lock, [this, job_ids] {
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
  finished_lock.unlock();
}

void Planner::WaitAll() {
  std::unique_lock<std::mutex> finished_lock(job_finished_mtx_);
  end_invoke_.wait(finished_lock, [this]() {
    return num_finished_jobs_ >= num_submitted_jobs_;
  });

  finished_lock.unlock();
}

void Planner::EnqueueFinishedJob(Job& job) {
  std::unique_lock<std::mutex> finished_lock(job_finished_mtx_);
  const bool is_finished =
      engine_.IsEnd(job.subgraph_key) || job.status != JobStatus::kSuccess;
  // record finished / failed job
  if (is_finished) {
    if (!job.is_idle_job) {
      jobs_finished_record_[GetJobRecordIndex(job.job_id)] = job;
      num_finished_jobs_++;
    }
    end_invoke_.notify_all();
  }
  // make sure to unlock before calling callback to avoid
  // potential recursive locking from client code
  finished_lock.unlock();

  // report end invoke using callback
  if (!job.is_idle_job) {
    if (on_end_request_ && job.require_callback && is_finished) {
      on_end_request_(job.job_id, job.status == JobStatus::kSuccess
                                      ? absl::OkStatus()
                                      : absl::InternalError("Job failed."));
    }
  }
}

void Planner::PrepareReenqueue(Job& job) {
  job.invoke_time = 0;
  job.end_time = 0;
  job.resolved_unit_subgraphs.clear();
  job.following_jobs.clear();
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
      break;
    }
    if (need_cpu_update_) {
      {
        auto status = SetCPUThreadAffinity(cpu_set_);
        if (!status.ok()) {
          BAND_LOG_PROD(BAND_LOG_WARNING, "%s", status.message());
        }
      }
      need_cpu_update_ = false;
    }
    BAND_LOG_PROD(BAND_LOG_INFO, "Planner triggered.");
    CopyToLocalQueues();
    BAND_LOG_PROD(BAND_LOG_INFO, "Planner copied to local queues.");
    bool need_reschedule = false;
    BAND_LOG_PROD(BAND_LOG_INFO, "Planner start scheduling.");
    need_reschedule |= !schedulers_[0]->Schedule(local_queue_);

    if (need_reschedule) {
      planner_safe_bool_.notify();
    }
  }
  return absl::OkStatus();
}

void Planner::CopyToLocalQueues() {
  std::unique_lock<std::mutex> request_lock(GetRequestsMtx());
  JobQueue& requests = GetRequests();
  if (!requests.empty()) {
    if (schedulers_.size() == 1) {
      // Gets jobs from requests and removes those jobs from the requests.
      auto& local_jobs = local_queue_;
      local_jobs.insert(local_jobs.end(),
                        std::make_move_iterator(requests.begin()),
                        std::make_move_iterator(requests.end()));
    }
    requests.clear();
  }
  request_lock.unlock();
}

bool Planner::EnqueueToWorker(const std::vector<ScheduleAction>& actions,
                              const std::vector<int> idle_uses) {
  bool success = true;
  for (int i = 0; i < actions.size(); i++) {
    Job job;
    SubgraphKey target_key;
    std::tie(job, target_key) = actions[i];

    Worker* worker = engine_.GetWorker(target_key.GetWorkerId());
    if (worker == nullptr) {
      BAND_LOG_PROD(BAND_LOG_ERROR,
                    "EnqueueToWorker failed. Requests scheduled to null worker "
                    "id %d",
                    target_key.GetWorkerId());
      job.status = JobStatus::kEnqueueFailed;
      EnqueueFinishedJob(job);
    } else if (IsSLOViolated(job)) {
      // no point in running this job anymore
      job.status = JobStatus::kSLOViolation;
      // mark this as -1 to differentiate it from the default value, 0
      job.invoke_time = -1;
      // mark the time of this decision (of early-dropping this job)
      job.end_time = time::NowMicros();
      // Set reschedule flag.
      success = false;
      EnqueueFinishedJob(job);
    } else {
      std::lock_guard<std::mutex> lock(worker->GetDeviceMtx());

      if (worker->IsEnqueueReady()) {
        UpdateJobScheduleStatus(job, target_key);
        if (!worker->EnqueueJob(job)) {
          BAND_LOG_PROD(BAND_LOG_ERROR,
                        "EnqueueToWorker failed. Requests scheduled to "
                        "unavailable worker id %d",
                        target_key.GetWorkerId());
        }
        if (idle_uses.size() > 0) {
          Job idle_job = Job::CreateIdleJob(idle_uses[i], target_key);
          if (!worker->EnqueueJob(idle_job)) {
            BAND_LOG_PROD(BAND_LOG_ERROR,
                          "EnqueueToWorker failed. Requests scheduled to "
                          "unavailable worker id %d",
                          target_key.GetWorkerId());
          }
        }
      } else {
        EnqueueRequest(job, true);
      }
    }
  }
  return success;
}

bool Planner::IsSLOViolated(Job& job) {
  if (job.status == JobStatus::kSLOViolation) {
    return true;
  }
  // this job has an SLO; check if it's not too late already
  if (job.slo_us > 0) {
    WorkerWaitingTime workers_waiting = engine_.GetWorkerWaitingTime();
    int64_t current_time = time::NowMicros();
    int64_t expected_latency = workers_waiting[job.subgraph_key.GetWorkerId()] +
                               job.expected_execution_time;
    int64_t remaining_time = job.slo_us - (current_time - job.enqueue_time);
    if (expected_latency > remaining_time) {
      return true;
    }
  }
  return false;
}

void Planner::UpdateJobScheduleStatus(Job& job, const SubgraphKey& target_key) {
  job.subgraph_key = target_key;
  job.sched_id = IssueSchedId();
  job.profiled_execution_time = engine_.GetProfiled(target_key);
  job.expected_execution_time = engine_.GetExpected(target_key);
  job.resolved_unit_subgraphs |= target_key.GetUnitIndices();

  if (!engine_.IsEnd(target_key)) {
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

}  // namespace band