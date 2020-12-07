#ifndef TENSORFLOW_LITE_WORKER_H_
#define TENSORFLOW_LITE_WORKER_H_

#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include "tensorflow/lite/core/cpu/cpu.h"

namespace tflite {

namespace impl {

class Planner;

struct Job {
  explicit Job(int model_id)
    : model_id_(model_id) {
  }
  int model_id_;
  int subgraph_idx_ = -1;
  int device_id_ = -1;
  int64_t enqueue_time_ = 0;
  int64_t invoke_time_ = 0;
  int64_t end_time_ = 0;
};

class Worker {
 public:
  explicit Worker(std::shared_ptr<Planner> planner);
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

  int SetCPUThreadAffinity(const CpuSet& thread_affinity_mask);

  const CpuSet& GetCpuSet() const { 
    return cpu_set_; 
  }

 private:
  void Work();

  std::weak_ptr<Planner> planner_;
  std::thread device_cpu_thread_;
  std::mutex device_mtx_;
  std::condition_variable request_cv_;
  std::deque<Job> requests_;
  bool kill_worker_ = false;
  CpuSet cpu_set_;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_WORKER_H_

