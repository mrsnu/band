#include "tensorflow/lite/worker.h"

#include <algorithm>

#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace impl {

bool GlobalQueueWorker::GiveJob(Job& job) {
  if (is_busy_ || !IsAvailable()) {
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

int GlobalQueueWorker::GetCurrentJobId() {
  std::unique_lock<std::mutex> lock(device_mtx_);
  return current_job_.job_id;
}

std::vector<thermal_t> GlobalQueueWorker::GetEstimatedEndTemperature() {
  std::unique_lock<std::mutex> lock(device_mtx_);
  // Return dummy values
  std::shared_ptr<Planner> planner_ptr = planner_.lock();
  return planner_ptr->GetResourceMonitor().GetAllTemperature();
}

int64_t GlobalQueueWorker::GetEstimatedFinishTime() {
  std::unique_lock<std::mutex> lock(device_mtx_);
  // Return dummy values
  return requests_.back().estimated_finish_time;
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
  if (!IsAvailable()) {
    return LARGE_WAITING_TIME;
  }

  if (!is_busy_) {
    return 0;
  }
  return 0;
}

void GlobalQueueWorker::Work() {
  while (true) {
    std::unique_lock<std::mutex> lock(device_mtx_);
    request_cv_.wait(lock, [this]() {
      return (kill_worker_ || is_busy_) && !is_paused_;
    });

    if (kill_worker_) {
      break;
    }

    lock.unlock();

    if (!IsValid(current_job_)) {
      TF_LITE_MAYBE_REPORT_ERROR(
          GetErrorReporter(),
          "%s worker failed to acquire ptr to planner",
          TfLiteDeviceGetName(device_flag_));
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
          interpreter_ptr->UpdateExpectedLatency(
              subgraph_idx,
              (current_job_.end_time - current_job_.invoke_time));
          if (current_job_.following_jobs.size() != 0) {
            planner_ptr->EnqueueBatch(current_job_.following_jobs);
          } 
          TryCopyOutputTensors(current_job_);
          current_job_.status = kTfLiteJobSuccess;

        } else if (status == kTfLiteDelegateError) {
          lock.lock();
          is_throttling_ = true;
          planner_ptr->PrepareReenqueue(current_job_);
          lock.unlock();

          planner_ptr->EnqueueRequest(current_job_, true);
          WaitUntilDeviceAvailable(subgraph);

          lock.lock();
          is_throttling_ = false;
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
        TF_LITE_MAYBE_REPORT_ERROR(
            GetErrorReporter(),
            "%s worker failed to copy input",
            TfLiteDeviceGetName(device_flag_));
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
      TF_LITE_MAYBE_REPORT_ERROR(
          GetErrorReporter(),
          "%s worker failed to acquire ptr to planner",
          TfLiteDeviceGetName(device_flag_));
      return;
    }
  }
}

}  // namespace impl
}  // namespace tflite
