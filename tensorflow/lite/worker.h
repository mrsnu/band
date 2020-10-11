#ifndef TENSORFLOW_LITE_WORKER_H_
#define TENSORFLOW_LITE_WORKER_H_

#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>

namespace tflite {

namespace impl {

class Planner;

struct Job {
  explicit Job(int model_id)
    : model_id_(model_id) {
    subgraph_idx_ = -1;
  }
  int model_id_;
  int subgraph_idx_;
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

