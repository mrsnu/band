#ifndef TENSORFLOW_LITE_WORKER_H_
#define TENSORFLOW_LITE_WORKER_H_

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

#include "tensorflow/lite/cpu.h"
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/config.h"
#include "tensorflow/lite/planner/util.h"

namespace tflite {

namespace impl {

class Planner;
class Subgraph;

class Worker {
 public:
  explicit Worker(std::shared_ptr<Planner> planner,
                  TfLiteDeviceFlags device_flag);
  virtual ~Worker();

  TfLiteStatus Init(WorkerConfig& config, int worker_id);
  TfLiteDeviceFlags GetDeviceFlag() const { return device_flag_; }
  std::mutex& GetDeviceMtx() { return device_mtx_; }
  std::condition_variable& GetRequestCv() { return request_cv_; }
  TfLiteStatus UpdateWorkerThread(const CpuSet thread_affinity_mask, int num_threads);
  void WaitUntilDeviceAvailable(Subgraph* subgraph);
  bool IsAvailable();
  void Pause();
  void Resume();
  const CpuSet& GetWorkerThreadAffinity() const;
  int GetNumThreads() const;

  worker_id_t GetId() {
    return worker_id_;
  }
  virtual int GetCurrentJobId() = 0;
  virtual int64_t GetWaitingTime() = 0;
  // Make sure the worker lock is acquired before calling the function.
  // Currently, `Planner::Plan()` is the only user of the method, and `Plan()` calls `GiveJob`
  // with the lock.
  virtual bool GiveJob(Job& job) = 0;


  // DeviceQueueWorker methods
  virtual JobQueue& GetDeviceRequests();

  // GlobalQueueWorker methods
  virtual bool IsBusy();

 protected:
  ErrorReporter* GetErrorReporter();
  bool IsValid(Job& job);
  TfLiteStatus TryCopyInputTensors(const Job& job);
  TfLiteStatus TryCopyOutputTensors(const Job& job);
  TfLiteStatus TryUpdateWorkerThread();
  virtual void Work() = 0;

  worker_id_t worker_id_ = -1;
  std::weak_ptr<Planner> planner_;
  std::thread device_cpu_thread_;
  std::mutex device_mtx_;
  std::condition_variable request_cv_;
  bool kill_worker_ = false;
  bool is_throttling_ = false;
  bool is_paused_ = false;
  
  // Configs
  int32_t availability_check_interval_ms_;
  std::string offloading_target_;
  int32_t offloading_data_size_;

  // GlobalQueueWorker doesn't actually use this for scheduling, but we
  // need this for the return value of GetDeviceRequests()
  JobQueue requests_;

  CpuSet cpu_set_;
  int num_threads_;
  bool need_cpu_update_ = false;
  std::mutex cpu_mtx_;

  const TfLiteDeviceFlags device_flag_;

  static const int64_t LARGE_WAITING_TIME = INT_MAX/2;
};

class DeviceQueueWorker : public Worker {
 public:
  explicit DeviceQueueWorker(std::shared_ptr<Planner> planner,
                             TfLiteDeviceFlags device_flag)
      : Worker(planner, device_flag) {
    device_cpu_thread_ = std::thread([this]{this->Work();});
  }

  int GetCurrentJobId() override;
  int64_t GetWaitingTime() override;
  bool GiveJob(Job& job) override;
  JobQueue& GetDeviceRequests() override;

 protected:
  void Work() override;

 private:
};

class GlobalQueueWorker : public Worker {
 public:
  explicit GlobalQueueWorker(std::shared_ptr<Planner> planner,
                             TfLiteDeviceFlags device_flag)
      : Worker(planner, device_flag) {
    device_cpu_thread_ = std::thread([this]{this->Work();});
  }

  int GetCurrentJobId() override;
  int64_t GetWaitingTime() override;
  bool GiveJob(Job& job) override;
  bool IsBusy() override;

 protected:
  void Work() override;

 private:
  Job current_job_{-1};
  bool is_busy_ = false;
};

class DeviceQueueOffloadingWorker : public Worker {
 public:
  explicit DeviceQueueOffloadingWorker(std::shared_ptr<Planner> planner,
                                       TfLiteDeviceFlags device_flag)
      : Worker(planner, device_flag) {
    device_cpu_thread_ = std::thread([this]{this->Work();});
  }

  std::thread periodic_thread_;
  int GetCurrentJobId() override;
  int64_t GetWaitingTime() override;
  bool GiveJob(Job& job) override;
  JobQueue& GetDeviceRequests() override;

 protected:
  void Work() override;
};

class GlobalQueueOffloadingWorker : public Worker {
 public:
  explicit GlobalQueueOffloadingWorker(std::shared_ptr<Planner> planner,
                             TfLiteDeviceFlags device_flag)
      : Worker(planner, device_flag) {
    device_cpu_thread_ = std::thread([this]{this->Work();});
  }

  int GetCurrentJobId() override;
  int64_t GetWaitingTime() override;
  bool GiveJob(Job& job) override;
  bool IsBusy() override;

 protected:
  void Work() override;

 private:
  Job current_job_{-1};
  bool is_busy_ = false;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_WORKER_H_
