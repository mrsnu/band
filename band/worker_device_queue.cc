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
      double invoke_time = (*it).invoke_time;
      if (invoke_time > 0 && current_time > invoke_time) {
        double progress = (current_time - invoke_time) > expected_latency
                               ? expected_latency
                               : (current_time - invoke_time);
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
  BAND_NOT_IMPLEMENTED;
}

}  // namespace band
