#ifndef TENSORFLOW_LITE_PLANNER_GLOBAL_QUEUE_PLANNER_H_
#define TENSORFLOW_LITE_PLANNER_GLOBAL_QUEUE_PLANNER_H_

#include <set>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/planner/planner.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace impl {

static bool compareJob(const Job& a, const Job& b) {
  return a.enqueue_time + a.slo < b.enqueue_time + b.slo;
  /*
  if (a.enqueue_time + a.slo != b.enqueue_time + b.slo) {
    return a.enqueue_time + a.slo != b.enqueue_time + b.slo;
  } else {
    return a.request_id < b.request_id;
  }*/
}

class GlobalQueuePlanner : public Planner {
 public:
  explicit GlobalQueuePlanner(Interpreter* interpreter)
      : Planner(interpreter) {
    planner_thread_ = std::thread([this]{this->Plan();});
  }

  void Plan() override;
  bool NeedProfile() override { return true; }
  void EnqueueRequest(Job job) override;
  void EnqueueBatch(std::vector<Job> jobs) override;

 private:
  std::set<Job, decltype(compareJob)*> ordered_requests_{compareJob};
  int total_num_jobs_ = 0;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_GLOBAL_QUEUE_PLANNER_H_
