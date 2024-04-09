#include "band/worker.h"

#include "absl/strings/str_format.h"
#include "band/common.h"
#include "band/job_tracer.h"
#include "band/logger.h"
#include "band/time.h"

namespace band {
Worker::Worker(IEngine* engine, WorkerId worker_id, DeviceFlag device_flag)
    : engine_(engine), worker_id_(worker_id), device_flag_(device_flag) {}

/**
 * @brief Destructor for the Worker class.
 * 
 * This destructor is responsible for cleaning up resources associated with the Worker object.
 * It checks if the worker thread has been explicitly stopped before destruction and logs an error message if not.
 */
Worker::~Worker() {
  if (!kill_worker_) {
    BAND_LOG(LogSeverity::kError,
             "Worker should explicitly stop worker thread before destruction");
  }
}

/**
 * @brief Initializes the Worker with the given configuration.
 * 
 * This function sets the availability check interval and logs the affinity of the worker.
 * It also updates the worker thread with the specified CPU mask and number of threads.
 * 
 * @param config The configuration for the Worker.
 * @return absl::Status The status of the initialization process.
 */
absl::Status Worker::Init(const WorkerConfig& config) {
  availability_check_interval_ms_ = config.availability_check_interval_ms;
  BAND_LOG_DEBUG("Set affinity of worker (%d,%s) to %s cores for %d threads.",
                 worker_id_, ToString(device_flag_),
                 ToString(config.cpu_masks[worker_id_]),
                 config.num_threads[worker_id_]);

  const CpuSet worker_mask_set =
      BandCPUMaskGetSet(config.cpu_masks[worker_id_]);
  return UpdateWorkerThread(worker_mask_set, config.num_threads[worker_id_]);
}

/**
 * @brief Updates the worker thread with the given thread affinity mask and number of threads.
 * 
 * This function updates the worker thread with the provided thread affinity mask and number of threads.
 * If the number of threads has changed, the function sets the `need_cpu_update_` flag to true.
 * 
 * @param thread_affinity_mask The CPU set representing the desired thread affinity mask.
 * @param num_threads The number of threads to be updated.
 * @return absl::Status The status of the update operation.
 * 
 * @note If the current CPU thread affinity cannot be retrieved, a warning message is logged and the function returns `absl::OkStatus()`.
 * @note If the current CPU thread affinity is already equal to the provided thread affinity mask or the number of enabled threads in the mask is 0, the function returns `absl::OkStatus()` without making any changes.
 * @note If the thread affinity mask is different from the current CPU thread affinity, the function updates the `cpu_set_` with the provided mask, sets the `need_cpu_update_` flag to true, and returns `absl::OkStatus()`.
 */
absl::Status Worker::UpdateWorkerThread(const CpuSet thread_affinity_mask,
                                        int num_threads) {
  std::lock_guard<std::mutex> cpu_lock(cpu_mtx_);

  if (num_threads_ != num_threads) {
    num_threads_ = num_threads;
    need_cpu_update_ = true;
  }

  CpuSet current_cpu_set;
  if (!GetCPUThreadAffinity(current_cpu_set).ok()) {
    // skip if not supports
    BAND_LOG(LogSeverity::kWarning,
             "Set affinity failed - not supported by the platform");
    return absl::OkStatus();
  }

  if (current_cpu_set == thread_affinity_mask ||
      thread_affinity_mask.NumEnabled() == 0) {
    return absl::OkStatus();
  }

  for (int cpu = 0; cpu < GetCPUCount(); cpu++) {
    if (cpu_set_.IsEnabled(cpu) != thread_affinity_mask.IsEnabled(cpu)) {
      cpu_set_ = thread_affinity_mask;
      need_cpu_update_ = true;
      return absl::OkStatus();
    }
  }
  return absl::OkStatus();
}

/**
 * Waits until the device is available for the given subgraph.
 * This function continuously checks the availability of the device by invoking the engine with the given subgraph.
 * It sleeps for a specified interval between each availability check.
 * Once the device becomes available, the function returns.
 *
 * @param subgraph The subgraph for which the device availability needs to be checked.
 */
void Worker::WaitUntilDeviceAvailable(SubgraphKey& subgraph) {
  while (true) {
    time::SleepForMicros(1000 * availability_check_interval_ms_);
    BAND_LOG(LogSeverity::kInternal, "Availability check at %d ms.",
             time::NowMicros());
    if (engine_->Invoke(subgraph).ok()) {
      return;
    }
  }
}

/**
 * Checks if the worker is available for processing.
 *
 * @return true if the worker is available, false otherwise.
 */
bool Worker::IsAvailable() const { return !is_throttling_ && !is_paused_; }

void Worker::Start() {
  std::call_once(device_cpu_start_flag_, [&]() {
    // 确保只执行一次
    device_cpu_thread_ = std::thread([this] { this->Work(); });
  });
}

void Worker::End() {
  {
    std::lock_guard<std::mutex> lock(device_mtx_);
    kill_worker_ = true;
  }
  request_cv_.notify_all();
  // 唤醒所有可能因为条件变量而阻塞的线程
  device_cpu_thread_.join();
  // 阻塞当前线程直到工作线程结束
}

void Worker::Pause() {
  std::lock_guard<std::mutex> lock(device_mtx_);
  is_paused_ = true;
}

void Worker::Resume() {
  std::unique_lock<std::mutex> lock(device_mtx_);
  is_paused_ = false;
  lock.unlock();

  request_cv_.notify_one();
  // 通过notify_one可能唤醒一个等待request_cv_的线程
}

void Worker::Wait() {
  std::unique_lock<std::mutex> lock(device_mtx_);
  wait_cv_.wait(lock, [&]() { return !HasJob(); });
  // 会阻塞当前线程，直到HasJob()为false。
  // HasJob是一个虚函数，预期在派生类中定义，用来检查是否还有未完成的任务。
}

const CpuSet& Worker::GetWorkerThreadAffinity() const { return cpu_set_; }

int Worker::GetNumThreads() const { return num_threads_; }

bool Worker::IsEnqueueReady() const { return IsAvailable(); }

bool Worker::IsValid(Job& job) {
  return job.model_id >= 0 && job.subgraph_key.IsValid() &&
         job.enqueue_time > 0 && job.invoke_time == 0 && job.end_time == 0;
}

/**
 * @brief Attempts to update the worker thread.
 * 
 * This function is responsible for updating the worker thread if necessary.
 * It acquires a lock on the CPU mutex and checks if a CPU update is needed.
 * If an update is needed, it performs the necessary operations to update the thread.
 * 
 * @return absl::Status The status of the operation.
 */
absl::Status Worker::TryUpdateWorkerThread() {
  std::lock_guard<std::mutex> cpu_lock(cpu_mtx_);
  if (need_cpu_update_) {
    need_cpu_update_ = false;

    // TODO: propagate num threads per each interpreter?
    // 是否需要在每个解释器中传播线程数？

    // Interpreter *interpreter_ptr = engine_ptr->GetModelExecutor();
    // 获取模型执行器的解释器指针
    // auto internal_backend =
    //     interpreter_ptr->GetCpuBackendContext()->internal_backend_context();
    // // 获取解释器的内部后端上下文
    // internal_backend->SetCpuSet(std::this_thread::get_id(), cpu_set_);
    // // 设置CPU集
    // internal_backend->SetMaxNumThreads(num_threads_);
    // // 设置最大线程数

#if BAND_IS_MOBILE
    if (cpu_set_.NumEnabled() == 0) {
      return absl::OkStatus();
    }

    if (!SetCPUThreadAffinity(cpu_set_).ok()) {
      return absl::InternalError(
          absl::StrFormat("Worker (%d, %s) failed to set cpu thread affinity",
                          worker_id_, ToString(device_flag_)));
    }
#endif
  }
  return absl::OkStatus();
}

/**
 * @brief Executes the work loop for the worker.
 *
 * This function is responsible for executing the work loop for the worker. 
 * The worker continuously checks for available jobs and performs the necessary operations to process them. 
 * It waits for a job to become available, and once a job is obtained, it checks if the job is valid. 
 * If the job is valid, it updates the worker thread, copies the input tensors, invokes the subgraph, and handles the output tensors. 
 * If the job is not valid, it logs an error and breaks out of the loop.
 *
 * @note This function is intended to be run in a separate thread.
 */
void Worker::Work() {
  while (true) {
    if (!HasJob()) {
      // 如果没有任务
      wait_cv_.notify_all();
      // 通知所有等待wait_cv_的线程
    }

    std::unique_lock<std::mutex> lock(device_mtx_);
    request_cv_.wait(
        lock, [this]() { return (kill_worker_ || HasJob()) && !is_paused_; });
        // 通过 request_cv_ 条件变量等待新的任务到来或者接收到终止工作的信号 kill_worker_。同时确保工作器没有被暂停 is_paused_

    if (kill_worker_) {
      break;
    }

    Job* current_job = GetCurrentJob();
    lock.unlock();

    if (!current_job || !IsValid(*current_job)) {
      // 如果当前任务为空或者任务无效
      BAND_LOG(LogSeverity::kError,
               "%s worker spotted an invalid job (model id %d, "
               "subgraph valid %d (%d, %d), "
               "enqueue time %d, invoke time %d, end time %d)",
               ToString(device_flag_), current_job->model_id,
               current_job->subgraph_key.IsValid(),
               current_job->subgraph_key.GetModelId(),
               current_job->subgraph_key.GetWorkerId(),
               current_job->enqueue_time, current_job->invoke_time,
               current_job->end_time);
      break;
    }

    SubgraphKey subgraph_key = current_job->subgraph_key;
    // 获取当前任务的子图键

    if (!TryUpdateWorkerThread().ok()) {
      // TODO #21: Handle errors in multi-thread environment
      // 在多线程环境中处理错误
      BAND_LOG(LogSeverity::kError, "Worker %d failed to update thread",
               worker_id_);
    }

    if (engine_->TryCopyInputTensors(*current_job).ok()) {
      lock.lock();
      current_job->invoke_time = time::NowMicros();
      // 记录任务的开始时间
      lock.unlock();

      BAND_TRACER_BEGIN_SUBGRAPH(*current_job);
      absl::Status status = engine_->Invoke(subgraph_key);
      if (status.ok()) {
        // end_time is never read/written by any other thread as long as
        // is_busy == true, so it's safe to update it w/o grabbing the lock
        // 只有在 is_busy 状态为 true 的情况下，end_time 变量才不会被其他线程读取或修改。
        // 因此，在这种情况下更新 end_time 而不加锁是安全的。
        current_job->end_time = time::NowMicros();
        engine_->UpdateLatency(
            subgraph_key, (current_job->end_time - current_job->invoke_time));
        if (current_job->following_jobs.size() != 0) {
          engine_->EnqueueBatch(current_job->following_jobs, true);
        }
        {
          auto status = engine_->TryCopyOutputTensors(*current_job);
          if (!status.ok()) {
            BAND_LOG(LogSeverity::kWarning, "%s", status.message());
          }
        }
        current_job->status = JobStatus::kSuccess;
      } else if (!status.ok()) {
        HandleDeviceError(*current_job);
        engine_->Trigger();
        BAND_LOG(LogSeverity::kError, "Worker %d failed to invoke job %d",
                 worker_id_, current_job->job_id);
        continue;
      } else {
        // end_time is never read/written by any other thread as long as
        // !requests_.empty(), so it's safe to update it w/o grabbing the lock
        current_job->end_time = time::NowMicros();
        // TODO #21: Handle errors in multi-thread environment
        current_job->status = JobStatus::kInvokeFailure;
      }
    } else {
      BAND_LOG(LogSeverity::kError, "Worker %d failed to copy input",
               worker_id_);
      // TODO #21: Handle errors in multi-thread environment
      current_job->status = JobStatus::kInputCopyFailure;
    }
    BAND_TRACER_END_SUBGRAPH(*current_job);
    engine_->EnqueueFinishedJob(*current_job);

    lock.lock();
    EndEnqueue();
    lock.unlock();

    engine_->Trigger();
    BAND_LOG(LogSeverity::kInternal, "Worker %d finished job %d", worker_id_,
             current_job->job_id);
  }
}

}  // namespace band