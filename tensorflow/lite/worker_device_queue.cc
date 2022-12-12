#include "tensorflow/lite/worker.h"

#include <algorithm>

#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/time.h"
#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "libtflite", __VA_ARGS__)
#include <android/log.h>
#else
#define LOGI(...) printf(__VA_ARGS__)
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

  std::shared_ptr<Planner> planner = planner_.lock();
  if (!planner) {
    return -1;
  }
  Interpreter* interpreter = planner->GetInterpreter();

  int64_t total = 0;
  for (JobQueue::iterator it = requests_.begin(); it != requests_.end(); ++it) {
    Subgraph* current_subgraph = interpreter->subgraph(it->subgraph_idx);
    int64_t expected_latency =
      planner->GetModelManager()->GetPredictedLatency(it->worker_id, current_subgraph);

    total += expected_latency;
    if (it == requests_.begin()) {
      int64_t current_time = profiling::time::NowMicros();
      int64_t invoke_time = (*it).invoke_time;
      if (invoke_time > 0 && current_time > invoke_time) {
        int64_t progress =
          (current_time - invoke_time) > expected_latency ? expected_latency
                                              : (current_time - invoke_time);
        total -= progress;
      }
    }
  }
  lock.unlock();

  return total;
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
      Subgraph* subgraph = interpreter_ptr->subgraph(subgraph_idx);

      if (TryUpdateWorkerThread() != kTfLiteOk) {
        // TODO #21: Handle errors in multi-thread environment
        break;
      }

      if (TryCopyInputTensors(current_job) == kTfLiteOk) {
        lock.lock();
        current_job.invoke_time = profiling::time::NowMicros();
        planner_ptr->GetResourceMonitor().FillJobInfoBefore(current_job);
        lock.unlock();

        TfLiteStatus status = subgraph->Invoke();
        if (status == kTfLiteOk) {
          planner_ptr->GetResourceMonitor().FillJobInfoAfter(current_job);
          current_job.end_time = profiling::time::NowMicros();
          current_job.latency = current_job.end_time - current_job.invoke_time;
          planner_ptr->GetModelManager()->Update(current_job, subgraph);

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
          // For not selecting this subgraph again
          planner_ptr->GetModelManager()->ProfileLatency(subgraph, 10000000); 
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
