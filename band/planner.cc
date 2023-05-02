#include "band/planner.h"

#include <fstream>

#include "band/context.h"
#include "band/job_tracer.h"
#include "band/logger.h"
#include "band/model_spec.h"
#include "band/scheduler/fixed_worker_scheduler.h"
#include "band/scheduler/heterogeneous_earliest_finish_time_scheduler.h"
#include "band/scheduler/least_slack_first_scheduler.h"
#include "band/scheduler/round_robin_scheduler.h"
#include "band/scheduler/shortest_expected_latency_scheduler.h"
#include "band/time.h"

#include "absl/strings/str_format.h"

namespace band {

Planner::Planner(Context& context) : num_submitted_jobs_(0), context_(context) {
  planner_thread_ = std::thread([this] {
    auto status = this->Plan();
    if (!status.ok()) {
      BAND_LOG_PROD(BAND_LOG_ERROR, "Planner thread failed: %s",
                    status.message());
    }
  });
}

Planner::~Planner() {
  if (log_path_.size()) {
    BAND_TRACER_DUMP(log_path_);
  }
  planner_safe_bool_.terminate();
  planner_thread_.join();
}

absl::Status Planner::Init(const PlannerConfig& config) {
  schedule_window_size_ = config.schedule_window_size;
  log_path_ = config.log_path;

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
    auto enqueue_time = time::NowMicros();
    for (auto& job : jobs) {
      REPORT_IF_JOB_METHOD_FAILS(job.Queued(num_submitted_jobs_++));
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
      if (jobs_finished_record_[GetJobRecordIndex(job_id)].id() != job_id) {
        return false;
      }
    }
    return true;
  });
  request_lock.unlock();
}

void Planner::WaitAll() {
  std::unique_lock<std::mutex> request_lock(requests_.mtx);
  end_invoke_.wait(request_lock, [this]() {
    return num_finished_jobs_ >= num_submitted_jobs_;
  });

  request_lock.unlock();
}

void Planner::EnqueueFinishedJob(Job& job) {
  std::lock_guard<std::mutex> request_lock(requests_.mtx);
  // record finished / failed job
  if (context_.IsEnd(job.subgraph_key()) || job.IsSuccess()) {
    jobs_finished_record_[GetJobRecordIndex(job.id())] = job;
    num_finished_jobs_++;
    end_invoke_.notify_all();
  }

  // report end invoke using callback
  if (on_end_request_ && job.require_callback() &&
      context_.IsEnd(job.subgraph_key())) {
    on_end_request_(job.id(), job.IsSuccess()
                                  ? absl::OkStatus()
                                  : absl::InternalError("Job failed."));
  }
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

absl::StatusOr<Job> Planner::GetFinishedJob(int job_id) {
  std::lock_guard<std::mutex> request_lock(requests_.mtx);
  if (!IsJobIdValid(job_id)) {
    return absl::InternalError("Job not found.");
  }
  if (jobs_finished_record_[GetJobRecordIndex(job_id)].id() == -1) {
    return absl::InternalError("Job not finished.");
  }
  return jobs_finished_record_[GetJobRecordIndex(job_id)];
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
      return absl::OkStatus();
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
    CopyToLocalQueues();
    do {
      need_reschedule_ = false;
      for (size_t i = 0; i < local_queues_.size(); ++i) {
        REPORT_IF_JOB_METHOD_FAILS(schedulers_[i]->Schedule(local_queues_[i]));
      }
    } while (need_reschedule_);
  }
  return absl::OkStatus();
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
        if (job.slo_us() > 0) {
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

    // TODO(widiba03304): check if SLO violation should be checked here
    if (IsSLOViolated(job)) {
      job.Error(JobStatus::ErrorState::SLOViolation, "SLO violation.");
      EnqueueFinishedJob(job);
      need_reschedule_ = true;
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
  if (job.status().error_state == JobStatus::ErrorState::SLOViolation) {
    return true;
  }
  // this job has an SLO; check if it's not too late already
  if (job.slo_us() > 0) {
    WorkerWaitingTime workers_waiting = context_.GetWorkerWaitingTime();
    int64_t current_time = time::NowMicros();
    int64_t expected_latency =
        workers_waiting[job.subgraph_key().GetWorkerId()] +
        job.expected_execution_time();

    if (current_time + expected_latency > job.enqueue_time() + job.slo_us()) {
      return true;
    }
  }
  return false;
}

void Planner::UpdateJobScheduleStatus(Job& job, const SubgraphKey& target_key) {
  REPORT_IF_JOB_METHOD_FAILS(job.AssignSubgraphKey(target_key));
  REPORT_IF_JOB_METHOD_FAILS(job.AssignSchedId(IssueSchedId()));
  REPORT_IF_JOB_METHOD_FAILS(job.UpdateProfileInfo(context_.GetProfiled(target_key),
                        context_.GetExpected(target_key)));
  REPORT_IF_JOB_METHOD_FAILS(job.UpdateSubgraphSchedule(context_));
}

bool Planner::IsJobIdValid(int job_id) {
  return num_submitted_jobs_ - job_id <= NUM_FINISHED_RECORDS;
}

int Planner::GetJobRecordIndex(int job_id) const {
  return job_id % NUM_FINISHED_RECORDS;
}

}  // namespace band
