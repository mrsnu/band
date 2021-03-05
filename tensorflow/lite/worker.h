#ifndef TENSORFLOW_LITE_WORKER_H_
#define TENSORFLOW_LITE_WORKER_H_

#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <string>
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
  // Average cpu frequency in khz
  int64_t average_freq_ = 0;
  std::string model_fname_;
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

  TfLiteStatus SetWorkerThreadAffinity(const CpuSet thread_affinity_mask);

 private:
  void Work();

  std::weak_ptr<Planner> planner_;
  std::thread device_cpu_thread_;
  std::mutex device_mtx_;
  std::condition_variable request_cv_;
  std::deque<Job> requests_;
  bool kill_worker_ = false;
  CpuSet cpu_set_;
  bool need_cpu_set_update_ = false;
  std::mutex cpu_set_mtx_;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_WORKER_H_

