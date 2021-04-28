#ifndef TENSORFLOW_LITE_WORKER_H_
#define TENSORFLOW_LITE_WORKER_H_

#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include "tensorflow/lite/cpu.h"
#include "tensorflow/lite/util.h"

namespace tflite {

namespace impl {

class Planner;

class Worker {
 public:
  explicit Worker(std::shared_ptr<Planner> planner, TfLiteDeviceFlags device_flag);
  ~Worker();

  std::mutex& GetDeviceMtx() {
    return device_mtx_;
  }

  std::condition_variable& GetRequestCv() {
    return request_cv_;
  }

  std::deque<Job>& GetDeviceRequests() {
    return requests_;
  }

  TfLiteStatus SetWorkerThreadAffinity(const CpuSet thread_affinity_mask);

  int64_t GetWaitingTime();

  void AllowWorkSteal() {
    allow_work_steal_ = true;
  }

 private:
  void Work();

  void TryWorkSteal();

  std::weak_ptr<Planner> planner_;
  std::thread device_cpu_thread_;
  std::mutex device_mtx_;
  std::condition_variable request_cv_;
  std::deque<Job> requests_;
  bool kill_worker_ = false;

  CpuSet cpu_set_;
  bool need_cpu_set_update_ = false;
  std::mutex cpu_set_mtx_;

  TfLiteDeviceFlags device_flag_;

  bool allow_work_steal_ = false;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_WORKER_H_

