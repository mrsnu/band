#ifndef TENSORFLOW_LITE_PLANNER_MULTI_LEVEL_QUEUE_PLANNER_H_
#define TENSORFLOW_LITE_PLANNER_MULTI_LEVEL_QUEUE_PLANNER_H_

#include "tensorflow/lite/planner/planner.h"
#include "tensorflow/lite/interpreter.h"

namespace tflite {

namespace impl {

class MultiLevelQueuePlanner : public Planner {
 public:
  explicit MultiLevelQueuePlanner(Interpreter* interpreter, int num_queues = 2)
      : Planner(interpreter) {
    if (num_queues <= 0) {
      TFLITE_LOG(ERROR) << "The number of queues must be larger than 0.";
      exit(-1);
    }
    planner_thread_ = std::thread([this]{this->Plan();});
    multi_level_queue_.resize(num_queues);
  }
  void Plan() override;
  bool NeedProfile() override;

  // Get the number of queues in the planner.
  size_t GetNumQueues() {
    return multi_level_queue_.size();
  }

 private:
  // Multi-level Queue.
  // Note that the `requests_` JobQueue in the planner is not part of
  // the JobQueues in the `multi_level_queue_`.
  // The the index is closer to 0, the higher the priority.
  std::vector<JobQueue> multi_level_queue_;

  // Move items in the queue to one-level lower queue, if certain condition
  // matches.
  void Demote();

  // Move items in the queue to one-level higher queue, if certain condition
  // matches.
  void Promote();

  void EnqueueJob(Job job, int queue_level);

  // Schedule the queue with the index `queue_level`.
  // If you want to apply different scheduling algorithm for different queues,
  // you can implement another scheduling algorithm.
  // Make sure the argument `queue_level` is valid before calling the method.
  void ScheduleQueue(size_t queue_level, DeviceWaitingTime& device_waiting);

  bool IsQueueLevelValid(size_t queue_level) {
    if (queue_level >= GetNumQueues()) return false;
    else return true;
  }
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_MULTI_LEVEL_QUEUE_PLANNER_H_
