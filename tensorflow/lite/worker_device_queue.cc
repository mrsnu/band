#include "tensorflow/lite/worker.h"

#include <algorithm>

#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace impl {

JobQueue& DeviceQueueWorker::GetDeviceRequests() {
  return requests_;
}

void DeviceQueueWorker::AllowWorkSteal() {
  allow_work_steal_ = true;
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
  for (JobQueue::iterator it = requests_.begin();
       it != requests_.end(); ++it) {
    Subgraph* current_subgraph = interpreter->subgraph(it->subgraph_idx);
    int64_t expected_latency =
      interpreter->GetExpectedLatency(it->subgraph_idx);

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
      Subgraph& subgraph = *(interpreter_ptr->subgraph(subgraph_idx));

      if (TryUpdateWorkerThread() != kTfLiteOk) {
        // TODO #21: Handle errors in multi-thread environment
        break;
      }

      if (TryCopyInputTensors(current_job) == kTfLiteOk) {
        lock.lock();
        current_job.invoke_time = profiling::time::NowMicros();
        lock.unlock();

        TfLiteStatus status = subgraph.Invoke();
        if (status == kTfLiteOk) {
          // end_time is never read/written by any other thread as long as
          // is_busy == true, so it's safe to update it w/o grabbing the lock
          current_job.end_time = profiling::time::NowMicros();
          interpreter_ptr->UpdateExpectedLatency(
              subgraph_idx,
              (current_job.end_time - current_job.invoke_time));
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
      bool empty = requests_.empty();
      lock.unlock();

      if (allow_work_steal_ && empty) {
        TryWorkSteal();
      }

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

void DeviceQueueWorker::TryWorkSteal() {
  // Note: Due to the removal of attribute `Worker::worker_id_`,
  // `target_worker->GetWorkerId() == worker_id_` should be updated to
  // `target_worker == this`, and the type of `max_latency_gain_worker` should
  // be changed from `int` to `Worker*`.

/*
  std::shared_ptr<Planner> planner_ptr = planner_.lock();
  if (!planner_ptr) {
    TFLITE_LOG(ERROR) << "Worker " << worker_id_
                      << " TryWorkSteal() Failed to acquire pointer to Planner";
    return;
  }

  Interpreter* interpreter_ptr = planner_ptr->GetInterpreter();
  int64_t max_latency_gain = -1;
  int max_latency_gain_worker = -1;
  int max_latency_gain_subgraph_idx = -1;
  for (auto& worker : interpreter_ptr->GetWorkers()) {
    Worker* target_worker = worker.get();
    if (target_worker->GetWorkerId() == worker_id_) {
      continue;
    }

    int64_t waiting_time = target_worker->GetWaitingTime();

    std::unique_lock<std::mutex> lock(target_worker->GetDeviceMtx());
    if (target_worker->GetDeviceRequests().size() < 2) {
      // There is nothing to steal here or
      // the job is being processed by the target worker,
      // so leave it alone.
      continue;
    }

    Job& job = target_worker->GetDeviceRequests().back();
    lock.unlock();

    Subgraph* orig_subgraph = interpreter_ptr->subgraph(job.subgraph_idx);
    SubgraphKey& orig_key = orig_subgraph->GetKey();
    SubgraphKey new_key(job.model_id, device_flag_, orig_key.input_ops,
                        orig_key.output_ops);
    int64_t expected_latency = interpreter_ptr->GetExpectedLatency(new_key);
    if (expected_latency == -1 || expected_latency > waiting_time) {
      // no point in stealing this job, it's just going to take longer
      continue;
    }

    int64_t latency_gain = waiting_time - expected_latency;
    if (latency_gain > max_latency_gain) {
      max_latency_gain = latency_gain;
      max_latency_gain_worker = target_worker->GetWorkerId();
      max_latency_gain_subgraph_idx = interpreter_ptr->GetSubgraphIdx(new_key);
    }
  }

  if (max_latency_gain < 0) {
    // no viable job to steal -- do nothing
    return;
  }

  Worker* target_worker = interpreter_ptr->GetWorker(max_latency_gain_worker);
  std::unique_lock<std::mutex> lock(target_worker->GetDeviceMtx(), std::defer_lock);
  std::unique_lock<std::mutex> my_lock(device_mtx_, std::defer_lock);
  std::lock(lock, my_lock);

  if (target_worker->GetDeviceRequests().empty()) {
    // target worker has went on and finished all of its jobs
    // while we were slacking off
    return;
  }

  // this must not be a reference,
  // otherwise the pop_back() below will invalidate it
  Job job = target_worker->GetDeviceRequests().back();
  if (job.invoke_time > 0) {
    // make sure the target worker hasn't started processing the job yet
    return;
  }

  if (!requests_.empty()) {
    // make sure that I still don't have any work to do
    return;
  }

  job.subgraph_idx = max_latency_gain_subgraph_idx;
  job.device_id = device_flag_;

  // finally, we perform the swap
  target_worker->GetDeviceRequests().pop_back();
  requests_.push_back(job);
*/
}

}  // namespace impl
}  // namespace tflite
