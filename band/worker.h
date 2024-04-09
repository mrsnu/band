#ifndef BAND_WORKER_H_
#define BAND_WORKER_H_

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

#include "band/config.h"
#include "band/engine_interface.h"
#include "band/device/cpu.h"

/**
 * @file worker.h
 * @brief Contains the declaration of the `band::Worker` class and its derived classes.
 */

namespace band {

/**
 * @class Worker
 * @brief Represents a worker that performs tasks on a specific device.
 *
 * The Worker class provides functionality for initializing, starting, pausing, and resuming
 * the worker, as well as enqueueing and processing jobs. It also provides methods for
 * retrieving information about the worker, such as its ID, device flag, and current job.
 *
 * The Worker class is an abstract base class, meaning that it cannot be instantiated directly.
 * Derived classes must implement the pure virtual functions `GetCurrentJobId()`,
 * `GetWaitingTime()`, `EnqueueJob()`, `IsEnqueueReady()`, and `HasJob()`.
 */
class Worker {
 public:
  /**
   * @brief Constructs a Worker object.
   * @param engine Pointer to the engine that the worker belongs to.
   * @param worker_id The ID of the worker.
   * @param device_flag The device flag indicating the type of device the worker is associated with.
   */
  explicit Worker(IEngine* engine, WorkerId worker_id, DeviceFlag device_flag);

  /**
   * @brief Destroys the Worker object.
   */
  virtual ~Worker();

  /**
   * @brief Initializes the worker with the given configuration.
   * @param config The configuration for the worker.
   * @return The status of the initialization.
   */
  absl::Status Init(const WorkerConfig& config);

  /**
   * @brief Gets the device flag of the worker.
   * @return The device flag.
   */
  DeviceFlag GetDeviceFlag() const;

  /**
   * @brief Gets the ID of the worker.
   * @return The worker ID.
   */
  WorkerId GetId() const;

  /**
   * @brief Gets the mutex for the device.
   * @return The mutex.
   */
  std::mutex& GetDeviceMtx();

  /**
   * @brief Gets the condition variable for requesting work.
   * @return The condition variable.
   */
  std::condition_variable& GetRequestCv();

  /**
   * @brief Updates the worker thread with the given thread affinity mask and number of threads.
   * @param thread_affinity_mask The thread affinity mask.
   * @param num_threads The number of threads.
   * @return The status of the update.
   */
  absl::Status UpdateWorkerThread(const CpuSet thread_affinity_mask, int num_threads);

  /**
   * @brief Waits until the device associated with the worker is available.
   * @param subgraph The subgraph key.
   */
  void WaitUntilDeviceAvailable(SubgraphKey& subgraph);

  /**
   * @brief Checks if the worker is available.
   * @return True if the worker is available, false otherwise.
   */
  bool IsAvailable() const;

  /**
   * @brief Starts the worker.
   */
  void Start();

  /**
   * @brief Ends the worker.
   */
  void End();

  /**
   * @brief Pauses the worker.
   */
  void Pause();

  /**
   * @brief Resumes the worker.
   */
  void Resume();

  /**
   * @brief Waits until the end of current requests.
   */
  void Wait();

  /**
   * @brief Gets the thread affinity of the worker.
   * @return The thread affinity.
   */
  const CpuSet& GetWorkerThreadAffinity() const;

  /**
   * @brief Gets the number of threads of the worker.
   * @return The number of threads.
   */
  int GetNumThreads() const;

  /**
   * @brief Gets the ID of the current job being processed by the worker.
   * @return The current job ID.
   */
  virtual int GetCurrentJobId() = 0;

  /**
   * @brief Gets the waiting time of the worker.
   * @return The waiting time.
   */
  virtual int64_t GetWaitingTime() = 0;

  /**
   * @brief Enqueues a job for the worker to process.
   * @param job The job to enqueue.
   * @return True if the job was successfully enqueued, false otherwise.
   */
  virtual bool EnqueueJob(Job& job) = 0;

  /**
   * @brief Checks if the worker is ready to enqueue a job.
   * @return True if the worker is ready to enqueue a job, false otherwise.
   */
  virtual bool IsEnqueueReady() const;

  /**
   * @brief Checks if the worker has a job.
   * @return True if the worker has a job, false otherwise.
   */
  virtual bool HasJob() = 0;

 protected:
  /**
   * @brief Checks if a job is valid.
   * @param job The job to check.
   * @return True if the job is valid, false otherwise.
   */
  bool IsValid(Job& job);

  /**
   * @brief Tries to update the worker thread.
   * @return The status of the update.
   */
  absl::Status TryUpdateWorkerThread();

  /**
   * @brief Performs the work of the worker.
   */
  void Work();

  /**
   * @brief Gets the current job being processed by the worker.
   * @return The current job.
   */
  virtual Job* GetCurrentJob() = 0;

  /**
   * @brief Ends the enqueue operation.
   */
  virtual void EndEnqueue() = 0;

  /**
   * @brief Handles a device error during job processing.
   * @param current_job The current job being processed.
   */
  virtual void HandleDeviceError(Job& current_job) = 0;

  IEngine* const engine_;  // Pointer to the engine that the worker belongs to.

  std::once_flag device_cpu_start_flag_;  // Flag for device CPU start.
  std::thread device_cpu_thread_;  // Thread for device CPU.
  mutable std::mutex device_mtx_;  // Mutex for the device.
  std::condition_variable request_cv_;  // Condition variable for requesting work.

  std::condition_variable wait_cv_;
  bool kill_worker_ = false;
  bool is_throttling_ = false;  // Indicates if the worker is throttling.
  bool is_paused_ = false;  // Indicates if the worker is paused.
  int availability_check_interval_ms_;  // Availability check interval.
  WorkerId worker_id_ = -1;  // Worker ID.

  CpuSet cpu_set_;
  int num_threads_;
  bool need_cpu_update_ = false;
  std::mutex cpu_mtx_;

  const DeviceFlag device_flag_;  // Device flag indicating the type of device the worker is associated with.

  static const int64_t LARGE_WAITING_TIME = std::numeric_limits<int>::max() / 2;  // Large waiting time.
};

/**
 * @class DeviceQueueWorker
 * @brief Represents a worker that uses a device queue for job processing.
 */
class DeviceQueueWorker : public Worker {
 public:
  /**
   * @brief Constructs a DeviceQueueWorker object.
   * @param engine Pointer to the engine that the worker belongs to.
   * @param worker_id The ID of the worker.
   * @param device_flag The device flag indicating the type of device the worker is associated with.
   */
  explicit DeviceQueueWorker(IEngine* engine, WorkerId worker_id, DeviceFlag device_flag);

  /**
   * @brief Gets the ID of the current job being processed by the worker.
   * @return The current job ID.
   */
  int GetCurrentJobId() override;

  /**
   * @brief Gets the waiting time of the worker.
   * @return The waiting time.
   */
  int64_t GetWaitingTime() override;

  /**
   * @brief Enqueues a job for the worker to process.
   * @param job The job to enqueue.
   * @return True if the job was successfully enqueued, false otherwise.
   */
  bool EnqueueJob(Job& job) override;

  /**
   * @brief Checks if the worker has a job.
   * @return True if the worker has a job, false otherwise.
   */
  bool HasJob() override;

  /**
   * @brief Gets the device requests queue.提供对内部任务队列的访问，允许外部查询和管理任务队列。
   * @return The device requests queue.
   */
  JobQueue& GetDeviceRequests();

  /**
   * @brief Allows work stealing for the worker.
   */
  void AllowWorkSteal();

 protected:
  /**
   * @brief Gets the current job being processed by the worker.
   * @return The current job.
   */
  Job* GetCurrentJob() override;

  /**
   * @brief Ends the enqueue operation.
   */
  void EndEnqueue() override;

  /**
   * @brief Handles a device error during job processing.
   * @param current_job The current job being processed.
   */
  void HandleDeviceError(Job& current_job) override;

 private:
  /**
   * @brief Tries to perform work stealing.实现了任务窃取逻辑
   */
  void TryWorkSteal();

  JobQueue requests_;  // Device requests queue.
  bool allow_work_steal_ = false;  // Indicates if work stealing is allowed.
};

/**
 * @class GlobalQueueWorker
 * @brief Represents a worker that uses a global queue for job processing.
 */
class GlobalQueueWorker : public Worker {
 public:
  /**
   * @brief Constructs a GlobalQueueWorker object.
   * @param engine Pointer to the engine that the worker belongs to.
   * @param worker_id The ID of the worker.
   * @param device_flag The device flag indicating the type of device the worker is associated with.
   */
  explicit GlobalQueueWorker(IEngine* engine, WorkerId worker_id, DeviceFlag device_flag);

  /**
   * @brief Gets the ID of the current job being processed by the worker.
   * @return The current job ID.
   */
  int GetCurrentJobId() override;

  /**
   * @brief Gets the waiting time of the worker.
   * @return The waiting time.
   */
  int64_t GetWaitingTime() override;

  /**
   * @brief Enqueues a job for the worker to process.
   * @param job The job to enqueue.
   * @return True if the job was successfully enqueued, false otherwise.
   */
  bool EnqueueJob(Job& job) override;

  /**
   * @brief Checks if the worker is ready to enqueue a job.
   * @return True if the worker is ready to enqueue a job, false otherwise.
   */
  bool IsEnqueueReady() const override;

  /**
   * @brief Checks if the worker has a job.
   * @return True if the worker has a job, false otherwise.
   */
  bool HasJob() override;

 protected:
  /**
   * @brief Gets the current job being processed by the worker.
   * @return The current job.
   */
  Job* GetCurrentJob() override;

  /**
   * @brief Ends the enqueue operation.
   */
  void EndEnqueue() override;

  /**
   * @brief Handles a device error during job processing.
   * @param current_job The current job being processed.
   */
  void HandleDeviceError(Job& current_job) override;

 private:
  Job current_job_{-1};  // Current job being processed.
  bool is_busy_ = false;  // Indicates if the worker is busy.
};

}  // namespace band

#endif  // BAND_WORKER_H_
