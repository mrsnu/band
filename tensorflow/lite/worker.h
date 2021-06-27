#ifndef TENSORFLOW_LITE_WORKER_H_
#define TENSORFLOW_LITE_WORKER_H_

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

#include "tensorflow/lite/cpu.h"
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/profiling/time.h"

namespace tflite {

namespace impl {

class Planner;

class Worker {
 public:
  explicit Worker(std::shared_ptr<Planner> planner,
                  TfLiteDeviceFlags device_flag);
  virtual ~Worker();

  std::mutex& GetDeviceMtx() { return device_mtx_; }
  std::condition_variable& GetRequestCv() { return request_cv_; }
  TfLiteStatus SetWorkerThreadAffinity(const CpuSet thread_affinity_mask);
  virtual int64_t GetWaitingTime() = 0;

  // DeviceQueueWorker methods
  virtual std::deque<Job>& GetDeviceRequests();
  virtual void AllowWorkSteal();

  // GlobalQueueWorker methods
  virtual bool GiveJob(Job& job);
  virtual bool IsBusy();

 protected:
  TfLiteStatus CopyInputTensors(const Job& job);
  TfLiteStatus CopyOutputTensors(const Job& job);
  TfLiteStatus ProcessJob(
      Job& job,
      std::function<void(Job&)> pre_process = [](Job& job) {
        job.process_time = profiling::time::NowMicros();
      },
      std::function<void(Job&)> pre_invoke = [](Job& job) {
        job.invoke_time = profiling::time::NowMicros();
      },
      std::function<void(Job&)> post_invoke = [](Job& job) {
        job.end_invoke_time = profiling::time::NowMicros();
      },
      std::function<void(Job&)> post_process = [](Job& job) {
        job.end_time = profiling::time::NowMicros();
      });
  virtual void Work() = 0;

  std::weak_ptr<Planner> planner_;
  std::thread device_cpu_thread_;
  std::mutex device_mtx_;
  std::condition_variable request_cv_;
  bool kill_worker_ = false;

  // GlobalQueueWorker doesn't actually use this for scheduling, but we
  // need this for the return value of GetDeviceRequests()
  std::deque<Job> requests_;

  CpuSet cpu_set_;
  bool need_cpu_set_update_ = false;
  std::mutex cpu_set_mtx_;

  TfLiteDeviceFlags device_flag_;
};

class DeviceQueueWorker : public Worker {
 public:
  explicit DeviceQueueWorker(std::shared_ptr<Planner> planner,
                             TfLiteDeviceFlags device_flag)
      : Worker(planner, device_flag) {
    device_cpu_thread_ = std::thread([this]{this->Work();});
  }

  int64_t GetWaitingTime() override;
  std::deque<Job>& GetDeviceRequests() override;
  void AllowWorkSteal() override;

 protected:
  void Work() override;

 private:
  void TryWorkSteal();

  bool allow_work_steal_ = false;
};

class GlobalQueueWorker : public Worker {
 public:
  explicit GlobalQueueWorker(std::shared_ptr<Planner> planner,
                             TfLiteDeviceFlags device_flag)
      : Worker(planner, device_flag) {
    device_cpu_thread_ = std::thread([this]{this->Work();});
  }

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
