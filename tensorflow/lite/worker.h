#ifndef TENSORFLOW_LITE_WORKER_H_
#define TENSORFLOW_LITE_WORKER_H_

#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>

namespace tflite {

namespace impl {

class Interpreter;
class Subgraph;
class Planner;

struct Job {
  explicit Job(int subgraph_idx)
    : subgraph_idx_(subgraph_idx) {}
  int subgraph_idx_;
};

class Worker {
 public:
  explicit Worker(std::shared_ptr<Planner> planner);
  ~Worker();

 private:
  friend class Planner;

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

