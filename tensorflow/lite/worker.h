#ifndef TENSORFLOW_LITE_WORKER_H_
#define TENSORFLOW_LITE_WORKER_H_

#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <map>
#include <string>

#include "tensorflow/lite/profiling/time.h"

namespace tflite {

namespace impl {

class Planner;

struct Job {
  explicit Job(int model_id)
    : model_id_(model_id) {}

  Job(int model_id, int slo_ms)
    : model_id_(model_id), slo_ms_(slo_ms) {}

  int model_id_;
  int subgraph_idx_ = -1;
  int device_id_ = -1;
  int64_t enqueue_time_ = 0;
  int64_t invoke_time_ = 0;
  int64_t end_time_ = 0;
  int slo_ms_ = 0;
  std::string model_fname;

  std::map<int, int64_t> waiting_time;
  std::map<int, int64_t> expected_latency;
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

  int64_t GetWaitingTime();

 private:
  void Work();

  std::weak_ptr<Planner> planner_;
  std::thread device_cpu_thread_;
  std::mutex device_mtx_;
  std::condition_variable request_cv_;
  std::deque<Job> requests_;
  bool kill_worker_ = false;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_WORKER_H_

