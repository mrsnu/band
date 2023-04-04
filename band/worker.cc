#include "band/worker.h"

#include "absl/strings/str_format.h"
#include "band/common.h"
#include "band/job_tracer.h"
#include "band/logger.h"
#include "band/time.h"

namespace band {
Worker::Worker(Context* context, WorkerId worker_id, DeviceFlags device_flag)
    : context_(context), worker_id_(worker_id), device_flag_(device_flag) {}

Worker::~Worker() {
  if (!kill_worker_) {
    BAND_LOG_ERROR(
        "Worker should explicitly stop worker thread before destruction");
  }
}

absl::Status Worker::Init(const WorkerConfig& config) {
  availability_check_interval_ms_ = config.availability_check_interval_ms;

  BAND_LOG_INFO("Set affinity of worker (%d,%s) to %s cores for %d threads.",
                worker_id_, GetName(device_flag_).c_str(),
                BandCPUMaskGetName(config.cpu_masks[worker_id_]),
                config.num_threads[worker_id_]);

  const CpuSet worker_mask_set =
      BandCPUMaskGetSet(config.cpu_masks[worker_id_]);
  return UpdateWorkerThread(worker_mask_set, config.num_threads[worker_id_]);
}

absl::Status Worker::UpdateWorkerThread(const CpuSet thread_affinity_mask,
                                        int num_threads) {
  std::lock_guard<std::mutex> cpu_lock(cpu_mtx_);

  if (num_threads_ != num_threads) {
    num_threads_ = num_threads;
    need_cpu_update_ = true;
  }

  CpuSet current_cpu_set;
  if (!GetCPUThreadAffinity(current_cpu_set).ok()) {
    // skip if not supports
    BAND_LOG_WARNING("Set affinity failed - not supported by the platform");
    return absl::OkStatus();
  }

  if (current_cpu_set == thread_affinity_mask ||
      thread_affinity_mask.NumEnabled() == 0) {
    return absl::OkStatus();
  }

  for (int cpu = 0; cpu < GetCPUCount(); cpu++) {
    if (cpu_set_.IsEnabled(cpu) != thread_affinity_mask.IsEnabled(cpu)) {
      cpu_set_ = thread_affinity_mask;
      need_cpu_update_ = true;
      return absl::OkStatus();
    }
  }
  return absl::OkStatus();
}

void Worker::WaitUntilDeviceAvailable(SubgraphKey& subgraph) {
  while (true) {
    Time::SleepForMicros(1000 * availability_check_interval_ms_);
    BAND_LOG_INFO("Availability check at %d ms.", Time::NowMicros());
    if (context_->Invoke(subgraph).ok()) {
      return;
    }
  }
}

bool Worker::IsAvailable() const { return !is_throttling_ && !is_paused_; }

void Worker::Start() {
  std::call_once(device_cpu_start_flag_, [&]() {
    device_cpu_thread_ = std::thread([this] { this->Work(); });
  });
}

void Worker::End() {
  {
    std::lock_guard<std::mutex> lock(device_mtx_);
    kill_worker_ = true;
  }
  request_cv_.notify_all();
  device_cpu_thread_.join();
}

void Worker::Pause() {
  std::lock_guard<std::mutex> lock(device_mtx_);
  is_paused_ = true;
}

void Worker::Resume() {
  std::unique_lock<std::mutex> lock(device_mtx_);
  is_paused_ = false;
  lock.unlock();

  request_cv_.notify_one();
}

void Worker::Wait() {
  std::unique_lock<std::mutex> lock(device_mtx_);
  wait_cv_.wait(lock, [&]() { return !HasJob(); });
}

const CpuSet& Worker::GetWorkerThreadAffinity() const { return cpu_set_; }

int Worker::GetNumThreads() const { return num_threads_; }

bool Worker::IsEnqueueReady() const { return IsAvailable(); }

const ErrorReporter* Worker::GetErrorReporter() const {
  return context_->GetErrorReporter();
}

bool Worker::IsValid(Job& job) {
  return job.model_id >= 0 && job.subgraph_key.IsValid() &&
         job.enqueue_time > 0 && job.invoke_time == 0 && job.end_time == 0;
}

absl::Status Worker::TryUpdateWorkerThread() {
  std::lock_guard<std::mutex> cpu_lock(cpu_mtx_);
  if (need_cpu_update_) {
    need_cpu_update_ = false;

    // TODO: propagate num threads per each interpreter?

    // Interpreter *interpreter_ptr = context_ptr->GetModelExecutor();
    // auto internal_backend =
    //     interpreter_ptr->GetCpuBackendContext()->internal_backend_context();
    // internal_backend->SetCpuSet(std::this_thread::get_id(), cpu_set_);
    // internal_backend->SetMaxNumThreads(num_threads_);

    if (cpu_set_.NumEnabled() == 0) {
      return absl::OkStatus();
    }

    if (!SetCPUThreadAffinity(cpu_set_).ok()) {
      return absl::InternalError(
          absl::StrFormat("Worker (%d, %s) failed to set cpu thread affinity",
                          worker_id_, GetName(device_flag_)));
    }
  }
  return absl::OkStatus();
}

void Worker::Work() {
  while (true) {
    std::unique_lock<std::mutex> lock(device_mtx_);

    if (!HasJob()) {
      wait_cv_.notify_all();
    }

    request_cv_.wait(
        lock, [this]() { return (kill_worker_ || HasJob()) && !is_paused_; });

    if (kill_worker_) {
      break;
    }

    Job* current_job = GetCurrentJob();
    lock.unlock();

    if (!current_job || !IsValid(*current_job)) {
      BAND_REPORT_ERROR(GetErrorReporter(),
                        "%s worker spotted an invalid job (model id %d, "
                        "subgraph valid %d (%d, %d), "
                        "enqueue time %d, invoke time %d, end time %d)",
                        GetName(device_flag_).c_str(), current_job->model_id,
                        current_job->subgraph_key.IsValid(),
                        current_job->subgraph_key.GetModelId(),
                        current_job->subgraph_key.GetWorkerId(),
                        current_job->enqueue_time, current_job->invoke_time,
                        current_job->end_time);
      break;
    }

    SubgraphKey subgraph_key = current_job->subgraph_key;

    if (!TryUpdateWorkerThread().ok()) {
      // TODO #21: Handle errors in multi-thread environment
      break;
    }

    if (context_->TryCopyInputTensors(*current_job).ok()) {
      lock.lock();
      current_job->invoke_time = Time::NowMicros();
      lock.unlock();

      BAND_TRACER_BEGIN_SUBGRAPH(*current_job);
      absl::Status status = context_->Invoke(subgraph_key);
      if (status.ok()) {
        // end_time is never read/written by any other thread as long as
        // is_busy == true, so it's safe to update it w/o grabbing the lock
        current_job->end_time = Time::NowMicros();
        context_->UpdateLatency(
            subgraph_key, (current_job->end_time - current_job->invoke_time));
        if (current_job->following_jobs.size() != 0) {
          context_->EnqueueBatch(current_job->following_jobs);
        }
        {
          auto status = context_->TryCopyOutputTensors(*current_job);
          if (!status.ok()) {
            BAND_LOG_WARNING("%s", status.message());
          }
        }
        current_job->status = JobStatus::Success;
      } else if (!status.ok()) {
        HandleDeviceError(*current_job);
        context_->Trigger();
        continue;
      } else {
        // end_time is never read/written by any other thread as long as
        // !requests_.empty(), so it's safe to update it w/o grabbing the lock
        current_job->end_time = Time::NowMicros();
        // TODO #21: Handle errors in multi-thread environment
        current_job->status = JobStatus::InvokeFailure;
      }
    } else {
      BAND_REPORT_ERROR(GetErrorReporter(), "%s worker failed to copy input",
                        GetName(device_flag_).c_str());
      // TODO #21: Handle errors in multi-thread environment
      current_job->status = JobStatus::InputCopyFailure;
    }
    BAND_TRACER_END_SUBGRAPH(*current_job);
    context_->EnqueueFinishedJob(*current_job);

    lock.lock();
    EndEnqueue();
    lock.unlock();

    context_->Trigger();
  }
}

}  // namespace band