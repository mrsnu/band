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

#include <algorithm>

#include "band/worker.h"

#include "band/common.h"
#include "band/logger.h"
#include "band/time.h"

namespace band {

bool GlobalQueueWorker::EnqueueJob(Job& job) {
  if (!IsEnqueueReady()) {
    BAND_LOG_PROD(BAND_LOG_ERROR, "Worker is not ready to enqueue");
    return false;
  }

  current_job_ = job;
  is_busy_ = true;
  request_cv_.notify_one();
  return true;
}

bool GlobalQueueWorker::IsEnqueueReady() const {
  return !is_busy_ && IsAvailable();
}

bool GlobalQueueWorker::HasJob() { return is_busy_; }

int GlobalQueueWorker::GetCurrentJobId() { return current_job_.job_id; }

Job* GlobalQueueWorker::GetCurrentJob() {
  return HasJob() ? &current_job_ : nullptr;
}

void GlobalQueueWorker::EndEnqueue() { is_busy_ = false; }

void GlobalQueueWorker::HandleDeviceError(Job& current_job) {
  std::unique_lock<std::mutex> lock(device_mtx_);
  lock.lock();
  is_throttling_ = true;
  engine_->PrepareReenqueue(current_job);
  lock.unlock();

  engine_->EnqueueRequest(current_job, true);
  WaitUntilDeviceAvailable(current_job.subgraph_key);

  lock.lock();
  is_throttling_ = false;
  is_busy_ = false;
  lock.unlock();
}

// This function returns the remaining time until this worker can start
// processing another Job.
//
// The remaining time is calculated based on the profiled model time of the
// Job, the timestamp of when this worker started processing the Job
// (current_job_.invoke_time), and the current timestamp.
// In case more time has passed (since invoke_time) than the profiled model
// time, this function returns 0, as it is unable to predict when the current
// job will finish.
// This function can also return 0 if the worker is not working on any job at
// the moment (HasJob() returns false).
//
// In case this function fails to acquire a shared ptr to the Planner,
// we print an error message and this function returns -1.
int64_t GlobalQueueWorker::GetWaitingTime() {
  std::unique_lock<std::mutex> lock(device_mtx_);
  if (!IsAvailable()) {
    return LARGE_WAITING_TIME;
  }

  if (!is_busy_) {
    return 0;
  }

  int64_t invoke_time = current_job_.invoke_time;

  // if this thread is the same thread that updates is_busy_ (false --> true)
  // and there are no other threads that call this function, then it is
  // technically safe to unlock here because the worker thread does not
  // update the other fields of current_job_
  // consider unlocking here if we need that teensy little extra perf boost
  // lock.unlock();

  // we no longer read from this worker's member variables, so there is
  // no need to hold on to the lock anymore
  lock.unlock();

  int64_t profiled_latency = engine_->GetExpected(current_job_.subgraph_key);

  if (invoke_time == 0) {
    // the worker has not started on processing the job yet
    return profiled_latency;
  }

  int64_t current_time = time::NowMicros();
  int64_t progress = current_time - invoke_time;
  return std::max((long)(profiled_latency - progress), 0L);
}

}  // namespace band
