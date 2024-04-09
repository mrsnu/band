#include <algorithm>

#include "band/common.h"
#include "band/logger.h"
#include "band/time.h"
#include "band/worker.h"
#include "worker.h"


/**
 * @brief The `GlobalQueueWorker` class represents a worker in the global job queue.
 * 
 * This class provides functionality to enqueue and process jobs in a global queue.
 * It also provides methods to check the status of the worker and retrieve information
 * about the current job being processed.
 */
namespace band {

/**
 * @brief Enqueues a job in the global queue for processing by this worker.
 * 
 * @param job The job to enqueue.
 * @return `true` if the job was successfully enqueued, `false` otherwise.
 */
bool GlobalQueueWorker::EnqueueJob(Job& job) {
  if (!IsEnqueueReady()) {
    BAND_LOG(LogSeverity::kError, "Worker is not ready to enqueue");
    return false;
  }

  current_job_ = job;
  is_busy_ = true;
  request_cv_.notify_one();
  return true;
}

/**
 * @brief Checks if the worker is ready to enqueue a job.
 * 
 * The worker is ready to enqueue a job if it is not currently busy and is available.
 * 
 * @return `true` if the worker is ready to enqueue a job, `false` otherwise.
 */
bool GlobalQueueWorker::IsEnqueueReady() const {
  return !is_busy_ && IsAvailable();
}

/**
 * @brief Checks if the worker has a job.
 * 
 * @return `true` if the worker has a job, `false` otherwise.
 */
bool GlobalQueueWorker::HasJob() { return is_busy_; }

/**
 * @brief Gets the ID of the current job being processed by the worker.
 * 
 * @return The ID of the current job being processed.
 */
int GlobalQueueWorker::GetCurrentJobId() { return current_job_.job_id; }

/**
 * @brief Gets a pointer to the current job being processed by the worker.
 * 
 * @return A pointer to the current job being processed, or `nullptr` if the worker
 *         is not currently processing any job.
 */
Job* GlobalQueueWorker::GetCurrentJob() {
  return HasJob() ? &current_job_ : nullptr;
}

/**
 * @brief Ends the enqueue process for the current job.
 * 
 * This method is called after the worker has finished processing the current job.
 * It sets the worker's busy flag to `false`.
 */
void GlobalQueueWorker::EndEnqueue() { is_busy_ = false; }

/**
 * @brief Handles a device error during job processing.
 * 
 * This method is called when a device error occurs during the processing of a job.
 * It prepares the job for re-enqueuing and then enqueues it again.
 * It also waits until the device becomes available before proceeding.
 * 
 * @param current_job The current job being processed.
 */
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
//此函数计算工作器开始处理下一个任务前的剩余时间。
// 
// The remaining time is calculated based on the profiled model time of the
// Job, the timestamp of when this worker started processing the Job
// (current_job_.invoke_time), and the current timestamp.
// In case more time has passed (since invoke_time) than the profiled model
// time, this function returns 0, as it is unable to predict when the current
// job will finish.
// This function can also return 0 if the worker is not working on any job at
// the moment (HasJob() returns false).
//这个剩余时间是根据任务的预设模型时间、工作器开始处理该任务的时间戳（current_job_.invoke_time）以及当前的时间戳来计算的。
// 如果从任务开始以来的实际耗时已超出预设的模型时间，该函数将返回0，表示无法预测当前任务何时完成。
// 如果工作器目前没有在处理任何任务（HasJob() 返回 false），函数同样会返回0。
// 
// In case this function fails to acquire a shared ptr to the Planner,
// we print an error message and this function returns -1.
// 如果无法获取到 Planner 的共享指针，将会打印错误信息，并且函数返回 -1。

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
  //如果这个线程是唯一一个更新 is_busy_ 状态（从 false 改为 true）的线程，并且没有其他线程执行这个函数，
  // 那么在这里解锁是技术上可行的，因为工作线程不会更改 current_job_ 的其他属性。
  // 如果需要获得微小的性能提升，可以考虑在这里解锁。lock.unlock(); 


  // we no longer read from this worker's member variables, so there is
  // no need to hold on to the lock anymore
  // 由于不再需要从该工作器的成员变量读取数据，保持锁定已经没有必要。
  lock.unlock();

  int64_t profiled_latency = engine_->GetExpected(current_job_.subgraph_key);

  if (invoke_time == 0) {
    // the worker has not started on processing the job yet
    //工作器尚未开始处理任务
    return profiled_latency;
  }

  int64_t current_time = time::NowMicros();
  int64_t progress = current_time - invoke_time;
  return std::max((long)(profiled_latency - progress), 0L);
}

}  // namespace band
