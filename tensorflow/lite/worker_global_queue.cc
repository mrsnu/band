#include "tensorflow/lite/worker.h"

#include <algorithm>

#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace impl {

bool GlobalQueueWorker::GiveJob(Job& job) {
  if (is_busy_ || !is_available_) {
    return false;
  }

  current_job_ = job;
  is_busy_ = true;
  request_cv_.notify_one();
  return true;
}

bool GlobalQueueWorker::IsBusy() {
  std::lock_guard<std::mutex> lock(device_mtx_);
  return is_busy_;
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
// the moment (IsBusy() returns false).
//
// In case this function fails to acquire a shared ptr to the Planner,
// we print an error message and this function returns -1.
int64_t GlobalQueueWorker::GetWaitingTime() {
  std::unique_lock<std::mutex> lock(device_mtx_);
  if (!is_available_) {
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

  std::shared_ptr<Planner> planner = planner_.lock();
  if (!planner) {
    TFLITE_LOG(ERROR) << "Worker " << device_flag_
                      << " failed to acquire ptr to planner";
    return -1;
  }
  Interpreter* interpreter = planner->GetInterpreter();

  // TODO #80: Get profiled_latency from current_job_
  Subgraph* current_subgraph = interpreter->subgraph(current_job_.subgraph_idx);
  int64_t profiled_latency =
      interpreter->GetExpectedLatency(current_job_.subgraph_idx);

  if (invoke_time == 0) {
    // the worker has not started on processing the job yet
    return profiled_latency;
  }

  int64_t current_time = profiling::time::NowMicros();
  int64_t progress = current_time - invoke_time;
  return std::max((long) (profiled_latency - progress), 0L);
}

void GlobalQueueWorker::Work() {
  while (true) {
    std::unique_lock<std::mutex> lock(device_mtx_);
    request_cv_.wait(lock, [this]() {
      return kill_worker_ || is_busy_;
    });

    if (kill_worker_) {
      break;
    }

    lock.unlock();

    if (!IsValid(current_job_)) {
      TFLITE_LOG(ERROR) << "Worker " << device_flag_
                        << " spotted an invalid job";
      break;
    }

    int subgraph_idx = current_job_.subgraph_idx;
    std::shared_ptr<Planner> planner_ptr = planner_.lock();
    if (planner_ptr) {
      Interpreter* interpreter_ptr = planner_ptr->GetInterpreter();
      Subgraph& subgraph = *(interpreter_ptr->subgraph(subgraph_idx));

      if (TryUpdateWorkerThread() != kTfLiteOk) {
        // TODO #21: Handle errors in multi-thread environment
        break;
      }

      if (TryCopyInputTensors(current_job_) == kTfLiteOk) {
        lock.lock();
        current_job_.invoke_time = profiling::time::NowMicros();
        lock.unlock();

        TfLiteStatus status = subgraph.Invoke();
        if (status == kTfLiteOk) {
          // end_time is never read/written by any other thread as long as
          // is_busy == true, so it's safe to update it w/o grabbing the lock
          current_job_.end_time = profiling::time::NowMicros();
          interpreter_ptr->UpdateInvokedLatency(
              subgraph_idx,
              (current_job_.end_time - current_job_.invoke_time),
              current_job_.start_target_frequency);
          if (current_job_.following_jobs.size() != 0) {
            planner_ptr->EnqueueBatch(current_job_.following_jobs);
          }
          TryCopyOutputTensors(current_job_);
          current_job_.status = kTfLiteJobSuccess;

        } else if (status == kTfLiteDelegateError) {
          lock.lock();
          is_available_ = false;
          planner_ptr->PrepareReenqueue(current_job_);
          lock.unlock();

          planner_ptr->EnqueueRequest(current_job_, true);
          WaitUntilDeviceAvailable(subgraph);

          lock.lock();
          is_available_ = true;
          is_busy_ = false;
          lock.unlock();

          planner_ptr->GetSafeBool().notify();
          continue;

        } else {
          // end_time is never read/written by any other thread as long as
          // is_busy == true, so it's safe to update it w/o grabbing the lock
          current_job_.end_time = profiling::time::NowMicros();
          // TODO #21: Handle errors in multi-thread environment
          current_job_.status = kTfLiteJobInvokeFailure;
        }
      } else {
        TFLITE_LOG(ERROR) << "Worker failed to copy input.";
        // TODO #21: Handle errors in multi-thread environment
        current_job_.status = kTfLiteJobInputCopyFailure;
      }
      planner_ptr->EnqueueFinishedJob(current_job_);

      lock.lock();
      is_busy_ = false;
      lock.unlock();

      planner_ptr->GetSafeBool().notify();
    } else {
      // TODO #21: Handle errors in multi-thread environment
      TFLITE_LOG(ERROR) << "Worker " << device_flag_
                        << " failed to acquire ptr to planner";
      return;
    }
  }
}

}  // namespace impl
}  // namespace tflite
