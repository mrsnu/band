#include "tensorflow/lite/planner/planner.h"

#include <fstream>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/planner/cloud_only_scheduler.h"
#include "tensorflow/lite/planner/mobile_only_heft_scheduler.h"
#include "tensorflow/lite/planner/mobile_cloud_heft_scheduler.h"
#include "tensorflow/lite/planner/random_assign_scheduler.h"
#include "tensorflow/lite/planner/thermal_aware_scheduler.h"
#include "tensorflow/lite/profiling/time.h"

#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "libtflite", __VA_ARGS__)
#include <android/log.h>
#endif // defined(__ANDROID__)

namespace tflite {
namespace impl {

Planner::Planner(Interpreter* interpreter) : interpreter_(interpreter), num_submitted_jobs_(0){
  planner_thread_ = std::thread([this] { this->Plan(); });
}

Planner::~Planner() {
  FlushFinishedJobs();
  planner_safe_bool_.terminate();
  planner_thread_.join();
  Finish();
}

TfLiteStatus Planner::Finish() {
  model_manager_->Close();
}

TfLiteStatus Planner::Init(PlannerConfig& config, ResourceConfig& resource_config) {
  schedule_window_size_ = config.schedule_window_size;
  log_path_ = config.log_path;

  // Init a ResourceMonitor instance
  if (resource_monitor_.Init(resource_config) != kTfLiteOk) {
    LOGI("ResourceMonitor init fails");
  }
  model_manager_ = new ModelManager(resource_monitor_);

  if (log_path_.size()) {
    std::ofstream log_file(log_path_);
    if (!log_file.is_open()) return kTfLiteError;
    log_file << "sched_id\t"
             << "model_name\t"
             << "model_id\t"
             << "worker_id\t"
             << "enqueue_time\t"
             << "invoke_time\t"
             << "end_time\t"
             << "estimated_latency\t"
             << "latency\t"
             << "communication_time\t"
             << "before_temp_cpu\t"
             << "before_temp_gpu\t"
             << "before_temp_dsp\t"
             << "before_temp_npu\t"
             << "before_temp_modem\t"
             << "frequency_cpu\t"
             << "frequency_gpu\t"
             << "estimated_temp\t"
             << "before_target_temp\t"
             << "after_target_temp\t"
             << "\t"
             << "est_ppt_cpu\t"
             << "est_ppt_gpu\t"
             << "est_ppt_dsp\t"
             << "est_ppt_npu\t"
             << "est_ppt_cloud\t"
             << "\t"
             << "est_temp_diff_cpu\t"
             << "est_temp_diff_gpu\t"
             << "est_temp_diff_dsp\t"
             << "est_temp_diff_npu\t"
             << "est_temp_diff_cloud\t"
             << "\t"
             << "est_total_latency_cpu\t"
             << "est_total_latency_gpu\t"
             << "est_total_latency_dsp\t"
             << "est_total_latency_npu\t"
             << "est_total_latency_cloud\t"
             << "job_status\n";
    log_file.close();
  }

  auto& schedulers = config.schedulers;
  if (schedulers.size() == 0 || schedulers.size() > 2) {
    TF_LITE_REPORT_ERROR(interpreter_->GetErrorReporter(),
                         "[Planner] Not supported for %d schedulers",
                         schedulers_.size());
    return kTfLiteError;
  }

  LOGI("Planner init");
  bool is_thermal_aware = false;
  local_queues_.resize(schedulers.size());
  for (int i = 0; i < schedulers.size(); ++i) {
    if (schedulers[i] == kCloudOnly) {
      schedulers_.emplace_back(new CloudOnlyScheduler(this));
    } else if (schedulers[i] == kMobileOnlyHeft) {
      schedulers_.emplace_back(new MobileOnlyHeftScheduler(this, model_manager_));
    } else if (schedulers[i] == kMobileCloudHeft) {
      schedulers_.emplace_back(new MobileCloudHeftScheduler(this, model_manager_));
    } else if (schedulers[i] == kRandomAssign) {
      schedulers_.emplace_back(new RandomAssignScheduler(this));
    } else if (schedulers[i] == kSplashHeft) {
      is_thermal_aware = true;
      schedulers_.emplace_back(new ThermalAwareScheduler(this, model_manager_));
    } else if (schedulers[i] == kSplashLst) {
      is_thermal_aware = true;
      schedulers_.emplace_back(new ThermalAwareScheduler(this, model_manager_));
    } else {
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

  if (model_manager_->Init(resource_config, is_thermal_aware) != kTfLiteOk) {
    LOGI("ModelManager init fails");
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

std::vector<std::unique_ptr<Worker>>& Planner::GetWorkers() {
  return interpreter_->GetWorkers();
}

bool Planner::NeedFallbackSubgraphs() const {
  return true;
}

void Planner::CopyToLocalQueues() {
  std::unique_lock<std::mutex> request_lock(GetRequestsMtx());
  JobQueue& requests = GetRequests();
  if (!requests.empty()) {
    // Gets jobs from requests and removes those jobs from the requests.
    auto& local_jobs = local_queues_[0];
    local_jobs.insert(local_jobs.end(),
                      std::make_move_iterator(requests.begin()),
                      std::make_move_iterator(requests.end()));
    requests.clear();
  }
  request_lock.unlock();
}

bool Planner::IsSLOViolated(Job& job) {
  if (job.status == kTfLiteJobSLOViolation) {
    return true;
  }
  // this job has an SLO; check if it's not too late already
  if (job.slo_us > 0) {
    int64_t current_time = profiling::time::NowMicros();
    int64_t expected_latency =
        workers_waiting_[job.worker_id] +
        job.estimated_latency;

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

std::set<int> Planner::GetIdleAllWorkers() {
  std::set<int> idle_workers;
  for (int i = 0; i < GetInterpreter()->GetNumWorkers(); ++i) {
    if (!GetInterpreter()->GetWorker(i)->IsBusy()) {
      idle_workers.insert(i);
    }
  }
  return idle_workers;
}

std::set<int> Planner::GetIdleAllWorkersForThermal() {
  std::set<int> idle_workers;
  for (int i = 0; i < GetInterpreter()->GetNumWorkers(); ++i) {
    if (i == kTfLiteCPU) {
      continue;
    }
    if (!GetInterpreter()->GetWorker(i)->IsBusy()) {
      idle_workers.insert(i);
    }
  }
  return idle_workers;
}


std::set<int> Planner::GetIdleProcessorWorkers() {
  std::set<int> idle_workers;
  for (int i = 0; i < GetInterpreter()->GetNumWorkers(); ++i) {
    if (i == kTfLiteCLOUD) {
      continue;
    }
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

  // report end invoke using callback
  if (on_end_invoke_ && interpreter_->subgraph(job.subgraph_idx)->IsEnd()) {
    on_end_invoke_(job.job_id,
                   job.status == kTfLiteJobSuccess ? kTfLiteOk : kTfLiteError);
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

void Planner::SetEndInvokeFunction(
  std::function<void(int, TfLiteStatus)> on_end_invoke) {
  on_end_invoke_ = on_end_invoke;
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

      if (job.slo_us > 0 && job.status == kTfLiteJobSuccess) {
        // check if slo has been violated or not
        auto latency = job.end_time - job.enqueue_time;
        job.status =
            latency > job.slo_us ? kTfLiteJobSLOViolation : kTfLiteJobSuccess;
      }
      // double ppt = 0.;
      // double fps = (double)(1000000. / (double)job.latency);
      // int temp_diff = job.after_target_temp[job.worker_id] - job.before_target_temp[job.worker_id];
      // if (temp_diff > 0) {
      // } else {
      //   temp_diff = 1;
      // }
      // ppt = fps / (double)(temp_diff) * 1000.;
      // write all timestamp statistics to log file
      log_file << job.sched_id << "\t"
               << job.model_fname << "\t"
               << job.model_id << "\t"
               << job.worker_id << "\t"
               << job.enqueue_time << "\t"
               << job.invoke_time << "\t"
               << job.end_time << "\t"
               << job.estimated_latency << "\t" 
               << job.latency << "\t"
               << job.communication_time << "\t"
               << job.before_temp[kTfLiteCPU] << "\t"
               << job.before_temp[kTfLiteGPU] << "\t"
               << job.before_temp[kTfLiteDSP] << "\t"
               << job.before_temp[kTfLiteNPU] << "\t"
               << job.before_temp[kTfLiteCLOUD] << "\t"
               << job.frequency[kTfLiteCPU] << "\t"
               << job.frequency[kTfLiteGPU] << "\t"
               << job.estimated_temp << "\t"
               << job.before_target_temp[job.worker_id] << "\t"
               << job.after_target_temp[job.worker_id] << "\t";
      log_file << "\t";
      if (job.estimated_ppt.size() != 0) {
        log_file << job.estimated_ppt[kTfLiteCPU] << "\t"
               << job.estimated_ppt[kTfLiteGPU] << "\t"
               << job.estimated_ppt[kTfLiteDSP] << "\t"
               << job.estimated_ppt[kTfLiteNPU] << "\t"
               << job.estimated_ppt[kTfLiteCLOUD] << "\t";
      }
      log_file << "\t";
      if (job.estimated_temp_diff.size() != 0) {
        log_file << job.estimated_temp_diff[kTfLiteCPU] << "\t"
               << job.estimated_temp_diff[kTfLiteGPU] << "\t"
               << job.estimated_temp_diff[kTfLiteDSP] << "\t"
               << job.estimated_temp_diff[kTfLiteNPU] << "\t"
               << job.estimated_temp_diff[kTfLiteCLOUD] << "\t";
      }
      log_file << "\t";
      if (job.estimated_total_latency.size() != 0) {
        log_file << job.estimated_total_latency[kTfLiteCPU] << "\t"
               << job.estimated_total_latency[kTfLiteGPU] << "\t"
               << job.estimated_total_latency[kTfLiteDSP] << "\t"
               << job.estimated_total_latency[kTfLiteNPU] << "\t"
               << job.estimated_total_latency[kTfLiteCLOUD] << "\t";
      }
      log_file << job.status << "\n";
    }
    log_file.close();
  } else {
    TF_LITE_REPORT_ERROR(interpreter_->GetErrorReporter(),
                         "[Planner] Invalid log file path %s",
                         log_path_.c_str());
  }
}

int Planner::IssueSchedId() { return sched_id_++; }

void Planner::UpdateJobScheduleStatus(Job& job, Subgraph* target_subgraph) {
  SubgraphKey& target_key = target_subgraph->GetKey();
  job.subgraph_idx = interpreter_->GetSubgraphIdx(target_key);
  job.worker_id = target_key.worker_id;
  job.device_id = interpreter_->GetWorkerDeviceFlag(target_key.worker_id);
  job.sched_id = IssueSchedId();
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
          if (GetInterpreter()->GetSubgraphIdx(model_id, worker_id) != -1) {
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
        TF_LITE_REPORT_ERROR(interpreter_->GetErrorReporter(),
                            "[Planner] Failed to set cpu thread affinity");
        // TODO #21: Handle errors in multi-thread environment
      } 
      need_cpu_update_ = false;
    }

    CopyToLocalQueues();
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
