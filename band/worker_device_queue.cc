#include <algorithm>

#include "band/logger.h"
#include "band/time.h"
#include "band/worker.h"

/**
 * @brief Namespace `band` contains classes and functions related to the band module.
 */
namespace band {

/**
 * @brief Get the reference to the job queue of the device queue worker.
 * @return Reference to the job queue.
 */
JobQueue& DeviceQueueWorker::GetDeviceRequests() { return requests_; }

/**
 * @brief Allow work steal for the device queue worker.
 */
void DeviceQueueWorker::AllowWorkSteal() { allow_work_steal_ = true; }

/**
 * @brief Check if the device queue worker has a job in the job queue.
 * @return True if the job queue is not empty, false otherwise.
 */
bool DeviceQueueWorker::HasJob() { return !requests_.empty(); }

/**
 * @brief Get the ID of the current job in the job queue.
 * @return The ID of the current job, or -1 if the job queue is empty.
 */
int DeviceQueueWorker::GetCurrentJobId() {
  if (requests_.empty()) {
    return -1;
  }
  return requests_.front().job_id;
}

/**
 * @brief Get the waiting time of the device queue worker.
 * @return The waiting time in microseconds.
 */
int64_t DeviceQueueWorker::GetWaitingTime() {
  std::unique_lock<std::mutex> lock(device_mtx_);
  if (!IsAvailable()) {
    return LARGE_WAITING_TIME;
  }

  int64_t total = 0;
  for (JobQueue::iterator it = requests_.begin(); it != requests_.end(); ++it) {
    int64_t expected_latency = engine_->GetExpected(it->subgraph_key);

    total += expected_latency;
    if (it == requests_.begin()) {
      int64_t current_time = time::NowMicros();
      int64_t invoke_time = (*it).invoke_time;
      if (invoke_time > 0 && current_time > invoke_time) {
        int64_t progress = (current_time - invoke_time) > expected_latency
                               ? expected_latency
                               : (current_time - invoke_time);
        total -= progress;
      }
    }
  }
  lock.unlock();

  return total;
}

/**
 * @brief Enqueue a job into the job queue of the device queue worker.
 * @param job The job to enqueue.
 * @return True if the job was successfully enqueued, false otherwise.
 */
bool DeviceQueueWorker::EnqueueJob(Job& job) {
  if (!IsEnqueueReady()) {
    return false;
  }
  requests_.push_back(job);
  request_cv_.notify_one();
  return true;
}

/**
 * @brief Get the pointer to the current job in the job queue.
 * @return Pointer to the current job, or nullptr if the job queue is empty.
 */
Job* DeviceQueueWorker::GetCurrentJob() {
  return HasJob() ? &requests_.front() : nullptr;
}

/**
 * @brief Remove the current job from the job queue and perform necessary actions.
 */
void DeviceQueueWorker::EndEnqueue() {
  requests_.pop_front();

  if (allow_work_steal_ && requests_.empty()) {
    // TODO: call this function once we re-implement
    // and test TryWorkSteal()
    // TryWorkSteal();
  }
}

/**
 * @brief Handle device error for the current job in the job queue.
 * @param current_job The current job.
 */
void DeviceQueueWorker::HandleDeviceError(Job& current_job) {
  std::unique_lock<std::mutex> lock(device_mtx_);
  lock.lock();
  is_throttling_ = true;
  engine_->PrepareReenqueue(current_job);
  std::vector<Job> jobs(requests_.begin(), requests_.end());
  requests_.clear();
  lock.unlock();

  engine_->EnqueueBatch(jobs, true);
  WaitUntilDeviceAvailable(current_job.subgraph_key);

  lock.lock();
  is_throttling_ = false;
  lock.unlock();
}

/**
 * @brief Try to steal work from other workers.
 */
void DeviceQueueWorker::TryWorkSteal() {
  // Note: Due to the removal of attribute `Worker::worker_id_`,
  // `target_worker->GetWorkerId() == worker_id_` should be updated to
  // `target_worker == this`, and the type of `max_latency_gain_worker` should
  // be changed from `int` to `Worker*`.
  // 注意：因为移除了 `Worker::worker_id_` 属性，
  // 原来的比较代码 `target_worker->GetWorkerId() == worker_id_` 需要更改为 `target_worker == this`。
  // 此外，`max_latency_gain_worker` 的数据类型也应从 `int` 转变为 `Worker*`。

  BAND_NOT_IMPLEMENTED;
  /*
    std::shared_ptr<Planner> planner_ptr = planner_.lock();
    if (!planner_ptr) {
      TFLITE_LOG(ERROR) << "Worker " << worker_id_
                        << " TryWorkSteal() Failed to acquire pointer to
    Planner"; return;
    }

    Interpreter* interpreter_ptr = planner_ptr->GetModelExecutor();
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
        max_latency_gain_subgraph_idx =
    interpreter_ptr->GetSubgraphIdx(new_key);
      }
    }

    if (max_latency_gain < 0) {
      // no viable job to steal -- do nothing
      return;
    }

    Worker* target_worker =
    interpreter_ptr->GetWorker(max_latency_gain_worker);
    std::unique_lock<std::mutex> lock(target_worker->GetDeviceMtx(),
    std::defer_lock); std::unique_lock<std::mutex> my_lock(device_mtx_,
    std::defer_lock); std::lock(lock, my_lock);

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

}  // namespace band
