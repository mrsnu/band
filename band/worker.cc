#include "band/worker.h"

#include "band/c/common.h"
#include "band/common.h"
#include "band/logger.h"
#include "band/time.h"

namespace Band {
Worker::Worker(Context* context, BandDeviceFlags device_flag)
    : device_flag_(device_flag), context_(context) {}

Worker::~Worker() {
  if (!kill_worker_) {
    BAND_LOG_INTERNAL(
        BAND_LOG_ERROR,
        "Worker should explicitly stop worker thread before destruction");
  }
}

BandStatus Worker::Init(const WorkerConfig& config, int worker_id) {
  if (config.allow_worksteal) {
    AllowWorkSteal();
  }

  availability_check_interval_ms_ = config.availability_check_interval_ms;
  worker_id_ = worker_id;

  BAND_LOG_INTERNAL(
      BAND_LOG_INFO,
      "Set affinity of worker (%d,%s) to %s cores for %d threads.", worker_id,
      BandDeviceGetName(device_flag_),
      BandCPUMaskGetName(config.cpu_masks[worker_id]),
      config.num_threads[worker_id]);

  const CpuSet worker_mask_set = BandCPUMaskGetSet(config.cpu_masks[worker_id]);
  return UpdateWorkerThread(worker_mask_set, config.num_threads[worker_id]);
}

BandStatus Worker::UpdateWorkerThread(const CpuSet thread_affinity_mask,
                                      int num_threads) {
  std::lock_guard<std::mutex> cpu_lock(cpu_mtx_);

  if (num_threads_ != num_threads) {
    num_threads_ = num_threads;
    need_cpu_update_ = true;
  }

  CpuSet current_cpu_set;
  if (GetCPUThreadAffinity(current_cpu_set) != kBandOk) {
    // skip if not supports
    BAND_LOG_INTERNAL(BAND_LOG_WARNING,
                      "Set affinity failed - not supported by the platform");
    return kBandOk;
  }

  if (current_cpu_set == thread_affinity_mask ||
      thread_affinity_mask.NumEnabled() == 0) {
    return kBandOk;
  }

  for (int cpu = 0; cpu < GetCPUCount(); cpu++) {
    if (cpu_set_.IsEnabled(cpu) != thread_affinity_mask.IsEnabled(cpu)) {
      cpu_set_ = thread_affinity_mask;
      need_cpu_update_ = true;
      return kBandOk;
    }
  }
  return kBandOk;
}

void Worker::WaitUntilDeviceAvailable(SubgraphKey& subgraph) {
  while (true) {
    Time::SleepForMicros(1000 * availability_check_interval_ms_);
    BAND_LOG_INTERNAL(BAND_LOG_INFO, "Availability check at %d ms.",
                      Time::NowMicros());
    if (context_->Invoke(subgraph) == kBandOk) {
      return;
    }
  }
}

bool Worker::IsAvailable() { return !is_throttling_ && !is_paused_; }

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

JobQueue& Worker::GetDeviceRequests() {
  JobQueue queue;
  BAND_NOT_IMPLEMENTED;
  return queue;
}

void Worker::AllowWorkSteal() { BAND_NOT_IMPLEMENTED; }

ErrorReporter* Worker::GetErrorReporter() {
  // TODO(dostos): return from context
  return context_->GetErrorReporter();
}

bool Worker::IsValid(Job& job) {
  return job.model_id >= 0 && job.subgraph_key.IsValid() &&
         job.enqueue_time > 0 && job.invoke_time == 0 && job.end_time == 0;
}

BandStatus Worker::TryUpdateWorkerThread() {
  std::lock_guard<std::mutex> cpu_lock(cpu_mtx_);
  if (need_cpu_update_) {
    need_cpu_update_ = false;

    // TODO: propagate num threads per each interpreter?

    // Interpreter *interpreter_ptr = context_ptr->GetInterpreter();
    // auto internal_backend =
    //     interpreter_ptr->GetCpuBackendContext()->internal_backend_context();
    // internal_backend->SetCpuSet(std::this_thread::get_id(), cpu_set_);
    // internal_backend->SetMaxNumThreads(num_threads_);

    if (cpu_set_.NumEnabled() == 0) {
      return kBandOk;
    }

    if (SetCPUThreadAffinity(cpu_set_) != kBandOk) {
      BAND_REPORT_ERROR(GetErrorReporter(),
                        "Worker (%d, %s) failed to set cpu thread affinity",
                        worker_id_, BandDeviceGetName(device_flag_));
      return kBandError;
    }
  }
  return kBandOk;
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
                        BandDeviceGetName(device_flag_), current_job->model_id,
                        current_job->subgraph_key.IsValid(),
                        current_job->subgraph_key.GetModelId(),
                        current_job->subgraph_key.GetWorkerId(),
                        current_job->enqueue_time, current_job->invoke_time,
                        current_job->end_time);
      break;
    }

    SubgraphKey subgraph_key = current_job->subgraph_key;

    if (TryUpdateWorkerThread() != kBandOk) {
      // TODO #21: Handle errors in multi-thread environment
      break;
    }

    if (context_->TryCopyInputTensors(*current_job) == kBandOk) {
      lock.lock();
      current_job->invoke_time = Time::NowMicros();
      lock.unlock();

      BandStatus status = context_->Invoke(subgraph_key);
      if (status == kBandOk) {
        // end_time is never read/written by any other thread as long as
        // is_busy == true, so it's safe to update it w/o grabbing the lock
        current_job->end_time = Time::NowMicros();
        context_->UpdateLatency(
            subgraph_key, (current_job->end_time - current_job->invoke_time));
        if (current_job->following_jobs.size() != 0) {
          context_->EnqueueBatch(current_job->following_jobs);
        }
        context_->TryCopyOutputTensors(*current_job);
        current_job->status = kBandJobSuccess;
      } else if (status == kBandDelegateError) {
        HandleDeviceError(*current_job);
        context_->Trigger();
        continue;
      } else {
        // end_time is never read/written by any other thread as long as
        // !requests_.empty(), so it's safe to update it w/o grabbing the lock
        current_job->end_time = Time::NowMicros();
        // TODO #21: Handle errors in multi-thread environment
        current_job->status = kBandJobInvokeFailure;
      }
    } else {
      BAND_REPORT_ERROR(GetErrorReporter(), "%s worker failed to copy input",
                        BandDeviceGetName(device_flag_));
      // TODO #21: Handle errors in multi-thread environment
      current_job->status = kBandJobInputCopyFailure;
    }
    context_->EnqueueFinishedJob(*current_job);

    lock.lock();
    EndEnqueue();
    lock.unlock();

    context_->Trigger();
  }
}

}  // namespace Band