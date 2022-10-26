#include "tensorflow/lite/worker.h"

#include <algorithm>

#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/time.h"
#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "libtflite", __VA_ARGS__)
#include <android/log.h>
#endif // defined(__ANDROID__)

namespace tflite {
namespace impl {

JobQueue& DeviceQueueWorker::GetDeviceRequests() {
  return requests_;
}

int DeviceQueueWorker::GetCurrentJobId() {
  std::unique_lock<std::mutex> lock(device_mtx_);
  if (requests_.empty()) {
    return -1;
  }
  return requests_.front().job_id;
}

int64_t DeviceQueueWorker::GetWaitingTime() {
  std::unique_lock<std::mutex> lock(device_mtx_);
  if (!IsAvailable()) {
    return LARGE_WAITING_TIME;
  }
  return 0;
}

bool DeviceQueueWorker::IsBusy() {
  std::unique_lock<std::mutex> lock(device_mtx_);
  return !requests_.empty();
}

std::vector<thermal_t> DeviceQueueWorker::GetEstimatedEndTemperature() {
  std::unique_lock<std::mutex> lock(device_mtx_);
  std::shared_ptr<Planner> planner_ptr = planner_.lock();
  std::vector<thermal_t> temp = planner_ptr->GetResourceMonitor().GetAllTemperature(); 
  if (requests_.empty()) {
    return temp;
  }
  temp[worker_id_] = requests_.back().estimated_temp;
  return temp;
}

int64_t DeviceQueueWorker::GetEstimatedFinishTime() {
  std::unique_lock<std::mutex> lock(device_mtx_);
  int64_t now = profiling::time::NowMicros();
  if (requests_.empty()) {
    return now;
  }
  int64_t expected = requests_.back().estimated_finish_time;
  if (now > expected) {
    return now;
  }
  // TODO : When preemption is available, this value could be wrong
  return expected;
}

bool DeviceQueueWorker::GiveJob(Job& job) {
  if (!IsAvailable()) {
    return false;
  }

  requests_.push_back(job);
  request_cv_.notify_one();
  return true;
}

void DeviceQueueWorker::Work() {
  while (true) {
    std::unique_lock<std::mutex> lock(device_mtx_);
    request_cv_.wait(lock, [this]() {
      return (kill_worker_ || !requests_.empty()) && !is_paused_;
    });

    if (kill_worker_) {
      break;
    }

    Job& current_job = requests_.front();
    lock.unlock();

    if (!IsValid(current_job)) {
      TF_LITE_MAYBE_REPORT_ERROR(
          GetErrorReporter(),
          "%s worker spotted an invalid job",
          TfLiteDeviceGetName(device_flag_));
      break;
    }

    int subgraph_idx = current_job.subgraph_idx;
    std::shared_ptr<Planner> planner_ptr = planner_.lock();
    if (planner_ptr) {
      Interpreter* interpreter_ptr = planner_ptr->GetInterpreter();
      Subgraph& subgraph = *(interpreter_ptr->subgraph(subgraph_idx));

      if (TryUpdateWorkerThread() != kTfLiteOk) {
        // TODO #21: Handle errors in multi-thread environment
        break;
      }

      if (TryCopyInputTensors(current_job) == kTfLiteOk) {
        lock.lock();
        current_job.invoke_time = profiling::time::NowMicros();
        current_job.before_temp = planner_ptr->GetResourceMonitor().GetAllTemperature();
        current_job.before_target_temp = planner_ptr->GetResourceMonitor().GetAllTargetTemperature();
        current_job.frequency = planner_ptr->GetResourceMonitor().GetAllFrequency(); 
        current_job.estimated_finish_time = current_job.invoke_time + current_job.estimated_latency; 
        lock.unlock();

        TfLiteStatus status = subgraph.Invoke();
        if (status == kTfLiteOk) {
          current_job.end_time = profiling::time::NowMicros();
          current_job.after_temp = planner_ptr->GetResourceMonitor().GetAllTemperature();
          current_job.after_target_temp = planner_ptr->GetResourceMonitor().GetAllTargetTemperature();
          current_job.latency = current_job.end_time - current_job.invoke_time;

          planner_ptr->GetModelManager()->Update(current_job);

          if (current_job.following_jobs.size() != 0) {
            planner_ptr->EnqueueBatch(current_job.following_jobs);
          } 
          TryCopyOutputTensors(current_job);
          current_job.status = kTfLiteJobSuccess;

        } else if (status == kTfLiteDelegateError) {
          lock.lock();
          is_throttling_ = true;
          planner_ptr->PrepareReenqueue(current_job);
          std::vector<Job> jobs(requests_.begin(), requests_.end());
          requests_.clear();
          lock.unlock();

          planner_ptr->EnqueueBatch(jobs, true);
          WaitUntilDeviceAvailable(subgraph);

          lock.lock();
          is_throttling_ = false;
          lock.unlock();

          planner_ptr->GetSafeBool().notify();
          continue;
        } else {
          // end_time is never read/written by any other thread as long as
          // !requests_.empty(), so it's safe to update it w/o grabbing the lock
          current_job.end_time = profiling::time::NowMicros();
          // TODO #21: Handle errors in multi-thread environment
          current_job.status = kTfLiteJobInvokeFailure;
        }
      } else {
        TF_LITE_MAYBE_REPORT_ERROR(
            GetErrorReporter(),
            "%s worker failed to copy input",
            TfLiteDeviceGetName(device_flag_));
        // TODO #21: Handle errors in multi-thread environment
        current_job.status = kTfLiteJobInputCopyFailure;
      }
      planner_ptr->EnqueueFinishedJob(current_job);
      
      lock.lock();
      requests_.pop_front();
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
