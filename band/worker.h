#ifndef BAND_WORKER_H_
#define BAND_WORKER_H_

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

#include "band/job.h"
#include "band/config.h"
#include "band/engine_interface.h"
#include "band/device/cpu.h"

namespace band {

class Planner;

static const int64_t LARGE_WAITING_TIME = std::numeric_limits<int>::max() / 2;

class Worker {
 public:
  explicit Worker(IEngine* engine, WorkerId worker_id,
                  DeviceFlag device_flag);
  virtual ~Worker();

  absl::Status Init(const WorkerConfig& config);
  DeviceFlag GetDeviceFlag() const { return device_flag_; }
  WorkerId GetId() const { return worker_id_; }
  std::mutex& GetDeviceMtx() { return device_mtx_; }
  std::condition_variable& GetRequestCv() { return request_cv_; }
  absl::Status UpdateWorkerThread(const CpuSet thread_affinity_mask,
                                  int num_threads);
  void WaitUntilDeviceAvailable(SubgraphKey& subgraph);
  bool IsAvailable() const;

  void Start();
  void End();

  void Pause();
  void Resume();
  // Wait until the end of current requests
  void Wait();

  const CpuSet& GetWorkerThreadAffinity() const;
  int GetNumThreads() const;
  virtual int GetCurrentJobId() = 0;
  virtual int64_t GetWaitingTime() = 0;
  // Make sure the worker lock is acquired before calling below functions.
  virtual bool EnqueueJob(Job& job) = 0;
  virtual bool IsEnqueueReady() const;
  virtual bool HasJob() = 0;

 protected:
  const ErrorReporter* GetErrorReporter() const;
  bool IsValid(Job& job);
  absl::Status TryUpdateWorkerThread();
  void Work();
  // Helper functions that work utilizes
  virtual Job* GetCurrentJob() = 0;
  virtual void EndEnqueue() = 0;
  virtual void HandleDeviceError(Job& current_job) = 0;

  IEngine* const engine_;

  std::once_flag device_cpu_start_flag_;
  std::thread device_cpu_thread_;
  mutable std::mutex device_mtx_;
  std::condition_variable request_cv_;
  std::condition_variable wait_cv_;
  bool kill_worker_ = false;
  bool is_throttling_ = false;
  bool is_paused_ = false;
  int availability_check_interval_ms_;
  WorkerId worker_id_ = -1;

  CpuSet cpu_set_;
  int num_threads_;
  bool need_cpu_update_ = false;
  std::mutex cpu_mtx_;

  const DeviceFlag device_flag_;
};

class DeviceQueueWorker : public Worker {
 public:
  explicit DeviceQueueWorker(IEngine* engine, WorkerId worker_id,
                             DeviceFlag device_flag)
      : Worker(engine, worker_id, device_flag) {}
  int GetCurrentJobId() override;
  int64_t GetWaitingTime() override;
  bool EnqueueJob(Job& job) override;
  bool HasJob() override;
  JobQueue& GetDeviceRequests();
  void AllowWorkSteal();

 protected:
  Job* GetCurrentJob() override;
  void EndEnqueue() override;
  void HandleDeviceError(Job& current_job) override;

 private:
  void TryWorkSteal();

  JobQueue requests_;
  bool allow_work_steal_ = false;
};

class GlobalQueueWorker : public Worker {
 public:
  explicit GlobalQueueWorker(IEngine* engine, WorkerId worker_id,
                             DeviceFlag device_flag)
      : Worker(engine, worker_id, device_flag) {}
  int GetCurrentJobId() override;
  int64_t GetWaitingTime() override;
  bool EnqueueJob(Job& job) override;
  bool IsEnqueueReady() const override;
  bool HasJob() override;

 protected:
  Job* GetCurrentJob() override;
  void EndEnqueue() override;
  void HandleDeviceError(Job& current_job) override;

 private:
  Job current_job_;
  bool is_busy_ = false;
};

}  // namespace band

#endif  // BAND_WORKER_H_
