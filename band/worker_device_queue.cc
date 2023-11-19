#include <algorithm>

#include "band/worker.h"

#include "band/logger.h"
#include "band/time.h"

namespace band {

JobQueue& DeviceQueueWorker::GetDeviceRequests() { return requests_; }

bool DeviceQueueWorker::HasJob() { return !requests_.empty(); }

int DeviceQueueWorker::GetCurrentJobId() {
  if (requests_.empty()) {
    return -1;
  }
  return requests_.front().job_id;
}

double DeviceQueueWorker::GetWaitingTime() {
  std::unique_lock<std::mutex> lock(device_mtx_);
  if (!IsAvailable()) {
    return LARGE_WAITING_TIME;
  }

  double total = 0;
  for (JobQueue::iterator it = requests_.begin(); it != requests_.end(); ++it) {
    double expected_latency = engine_->GetExpected(it->subgraph_key);

    total += expected_latency;
    if (it == requests_.begin()) {
      double current_time = time::NowMicros();
      double start_time = (*it).start_time;
      if (start_time > 0 && current_time > start_time) {
        double progress = (current_time - start_time) > expected_latency
                               ? expected_latency
                               : (current_time - start_time);
        total -= progress;
      }
    }
  }
  lock.unlock();

  return total;
}

bool DeviceQueueWorker::EnqueueJob(Job& job) {
  if (!IsEnqueueReady()) {
    return false;
  }

  BAND_LOG_PROD(BAND_LOG_INFO, "Enqueue job %d to worker %d", job.job_id,
                worker_id_);

  requests_.push_back(job);
  request_cv_.notify_one();
  return true;
}

Job* DeviceQueueWorker::GetCurrentJob() {
  return HasJob() ? &requests_.front() : nullptr;
}

void DeviceQueueWorker::EndEnqueue() {
  requests_.pop_front();

  if (allow_work_steal_ && requests_.empty()) {
    // TODO: call this function once we re-implement
    // and test TryWorkSteal()
    // TryWorkSteal();
  }
}

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

void DeviceQueueWorker::TryWorkSteal() {
  // Note: Due to the removal of attribute `Worker::worker_id_`,
  // `target_worker->GetWorkerId() == worker_id_` should be updated to
  // `target_worker == this`, and the type of `max_latency_gain_worker` should
  // be changed from `int` to `Worker*`.

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
    if (job.start_time > 0) {
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
