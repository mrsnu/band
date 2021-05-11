#ifndef TENSORFLOW_LITE_WORKER_H_
#define TENSORFLOW_LITE_WORKER_H_

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

#include "tensorflow/lite/cpu.h"
#include "tensorflow/lite/util.h"

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

  // WorkerDeviceQueue methods
  virtual std::deque<Job>& GetDeviceRequests() = 0;
  virtual void AllowWorkSteal() = 0;

  // WorkerGlobalQueue methods
  virtual bool GiveJob(Job& job) = 0;
  virtual bool IsBusy() = 0;

 protected:
  virtual void Work() = 0;

  std::weak_ptr<Planner> planner_;
  std::thread device_cpu_thread_;
  std::mutex device_mtx_;
  std::condition_variable request_cv_;
  bool kill_worker_ = false;

  CpuSet cpu_set_;
  bool need_cpu_set_update_ = false;
  std::mutex cpu_set_mtx_;

  TfLiteDeviceFlags device_flag_;
};

class WorkerDeviceQueue : public Worker {
 public:
  explicit WorkerDeviceQueue(std::shared_ptr<Planner> planner,
                             TfLiteDeviceFlags device_flag)
      : Worker(planner, device_flag) {}

  int64_t GetWaitingTime() override;

  // WorkerDeviceQueue methods
  std::deque<Job>& GetDeviceRequests() override;
  void AllowWorkSteal() override;

  // WorkerGlobalQueue methods
  bool GiveJob(Job& job) override;
  bool IsBusy() override;

 protected:
  void Work() override;

 private:
  void TryWorkSteal();

  std::deque<Job> requests_;
  bool allow_work_steal_ = false;
};

class WorkerGlobalQueue : public Worker {
 public:
  explicit WorkerGlobalQueue(std::shared_ptr<Planner> planner,
                             TfLiteDeviceFlags device_flag)
      : Worker(planner, device_flag) {}

  int64_t GetWaitingTime() override;

  // WorkerDeviceQueue methods
  std::deque<Job>& GetDeviceRequests() override;
  void AllowWorkSteal() override;

  // WorkerGlobalQueue methods
  bool GiveJob(Job& job) override;
  bool IsBusy() override;

 protected:
  void Work() override;

 private:
  Job current_job_{-1};
  bool is_busy_ = false;

  // we doesn't actually use this
  // this was put here solely for GetDeviceRequests()
  std::deque<Job> requests_dummy_;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_WORKER_H_

