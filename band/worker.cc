// Copyright 2023 Seoul National University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "band/worker.h"

#include "absl/strings/str_format.h"
#include "band/common.h"
#include "band/job_tracer.h"
#include "band/logger.h"
#include "band/time.h"

namespace band {
Worker::Worker(IEngine* engine, WorkerId worker_id, DeviceFlag device_flag)
    : engine_(engine), worker_id_(worker_id), device_flag_(device_flag) {}

Worker::~Worker() {
  if (!kill_worker_) {
    BAND_LOG(LogSeverity::kError,
             "Worker should explicitly stop worker thread before destruction");
  }
}

absl::Status Worker::Init(const WorkerConfig& config) {
  availability_check_interval_ms_ = config.availability_check_interval_ms;
  BAND_LOG_DEBUG("Set affinity of worker (%d,%s) to %s cores for %d threads.",
                 worker_id_, ToString(device_flag_),
                 ToString(config.cpu_masks[worker_id_]),
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
    BAND_LOG(LogSeverity::kWarning,
             "Set affinity failed - not supported by the platform");
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
    time::SleepForMicros(1000 * availability_check_interval_ms_);
    BAND_LOG(LogSeverity::kInternal, "Availability check at %d ms.",
             time::NowMicros());
    if (engine_->Invoke(subgraph).ok()) {
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

bool Worker::IsValid(Job& job) {
  return job.model_id >= 0 && job.subgraph_key.IsValid() &&
         job.enqueue_time > 0 && job.invoke_time == 0 && job.end_time == 0;
}

absl::Status Worker::TryUpdateWorkerThread() {
  std::lock_guard<std::mutex> cpu_lock(cpu_mtx_);
  if (need_cpu_update_) {
    need_cpu_update_ = false;

    // TODO: propagate num threads per each interpreter?

    // Interpreter *interpreter_ptr = engine_ptr->GetModelExecutor();
    // auto internal_backend =
    //     interpreter_ptr->GetCpuBackendContext()->internal_backend_context();
    // internal_backend->SetCpuSet(std::this_thread::get_id(), cpu_set_);
    // internal_backend->SetMaxNumThreads(num_threads_);

#if BAND_IS_MOBILE
    if (cpu_set_.NumEnabled() == 0) {
      return absl::OkStatus();
    }

    if (!SetCPUThreadAffinity(cpu_set_).ok()) {
      return absl::InternalError(
          absl::StrFormat("Worker (%d, %s) failed to set cpu thread affinity",
                          worker_id_, ToString(device_flag_)));
    }
#endif
  }
  return absl::OkStatus();
}

void Worker::Work() {
  while (true) {
    if (!HasJob()) {
      wait_cv_.notify_all();
    }

    std::unique_lock<std::mutex> lock(device_mtx_);
    request_cv_.wait(
        lock, [this]() { return (kill_worker_ || HasJob()) && !is_paused_; });

    if (kill_worker_) {
      break;
    }

    Job* current_job = GetCurrentJob();
    lock.unlock();

    if (!current_job || !IsValid(*current_job)) {
      BAND_LOG(LogSeverity::kError,
               "%s worker spotted an invalid job (model id %d, "
               "subgraph valid %d (%d, %d), "
               "enqueue time %d, invoke time %d, end time %d)",
               ToString(device_flag_), current_job->model_id,
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
      BAND_LOG(LogSeverity::kError, "Worker %d failed to update thread",
               worker_id_);
    }

    if (engine_->TryCopyInputTensors(*current_job).ok()) {
      lock.lock();
      current_job->invoke_time = time::NowMicros();
      lock.unlock();

      BAND_TRACER_BEGIN_SUBGRAPH(*current_job);
      absl::Status status = engine_->Invoke(subgraph_key);
      if (status.ok()) {
        // end_time is never read/written by any other thread as long as
        // is_busy == true, so it's safe to update it w/o grabbing the lock
        current_job->end_time = time::NowMicros();
        engine_->UpdateLatency(
            subgraph_key, (current_job->end_time - current_job->invoke_time));
        if (current_job->following_jobs.size() != 0) {
          engine_->EnqueueBatch(current_job->following_jobs, true);
        }
        {
          auto status = engine_->TryCopyOutputTensors(*current_job);
          if (!status.ok()) {
            BAND_LOG(LogSeverity::kWarning, "%s", status.ToString().c_str());
          }
        }
        current_job->status = JobStatus::kSuccess;
      } else if (!status.ok()) {
        HandleDeviceError(*current_job);
        engine_->Trigger();
        BAND_LOG(LogSeverity::kError, "Worker %d failed to invoke job %d",
                 worker_id_, current_job->job_id);
        continue;
      } else {
        // end_time is never read/written by any other thread as long as
        // !requests_.empty(), so it's safe to update it w/o grabbing the lock
        current_job->end_time = time::NowMicros();
        // TODO #21: Handle errors in multi-thread environment
        current_job->status = JobStatus::kInvokeFailure;
      }
    } else {
      BAND_LOG(LogSeverity::kError, "Worker %d failed to copy input",
               worker_id_);
      // TODO #21: Handle errors in multi-thread environment
      current_job->status = JobStatus::kInputCopyFailure;
    }
    BAND_TRACER_END_SUBGRAPH(*current_job);
    engine_->EnqueueFinishedJob(*current_job);

    lock.lock();
    EndEnqueue();
    lock.unlock();

    engine_->Trigger();
    BAND_LOG(LogSeverity::kInternal, "Worker %d finished job %d", worker_id_,
             current_job->job_id);
  }
}

}  // namespace band