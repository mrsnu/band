#include "tensorflow/lite/planner/two_level_planner.h"
#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace impl {

void MultiLevelQueuePlanner::Plan() {
  while (true) {
    if (GetSafeBool().wait())
      return;

    // Move all the jobs to the queue with the highest priority.
    CopyToLocalQueue(multi_level_queue_[0]);
    // Move the jobs according to the decision function.
    // There can be a situation where the same job is demoted and promoted
    // at the same time. Programmer should be careful when implementing the
    // decision functions.
    // Or we can promote jobs after scheduling to avoid such cases.
    // `decide_demote` may decide if to demote considering the indicated SLO.
    // `decide_promote` may decide if to promote considering the remaing deadline.
    Demote(decide_demote);
    Promote(decide_promote);

    // Update the worker status.
    UpdateDeviceWaitingTime();

    // Schedule each queue starting from the queue with the highest priority.
    for (size_t i = 0; i < GetNumQueues(); ++i) {
      if (IsQueueLevelValid(i)) {
        ScheduleQueue(i, device_waiting);
      } else {
        TFLITE_LOG(WARN) << "The selected queue level is invalid.";
      }
    }
  }
}

void MultiLevelQueuePlanner::Demote(DecisionFn decide_demote) {
  // Note that there is no more queue to demote for the last queue.
  for (int queue_level = 0; queue_level < GetNumQueues() - 1; ++queue_level) {
    auto& current_queue = multi_level_queue_[queue_level];
    for (auto job_it = current_queue.begin(); job_it != current_queue.end();) {
      if (decide_demote(job_it, device_waiting_, queue_level)) {
        EnqueueJob(*job_it, queue_level + 1);
        job_it = current_queue.erase(job_it);
      } else {
        ++job_it;
      }
    }
  }
}

void MultiLevelQueuePlanner::Promote(DecisionFn decide_promote) {
  // Note that there is no more queue to promote for the first queue.
  for (int queue_level = GetNumQueues() - 1; queue_level > 0; --queue_level) {
    auto& current_queue = multi_level_queue_[queue_level];
    for (auto job_it = current_queue.begin(); job_it != current_queue.end();) {
      if (decide_promote(job_it, device_waiting_, queue_level)) {
        EnqueueJob(*job_it, queue_level - 1);
        job_it = current_queue.erase(job_it);
      } else {
        ++job_it;
      }
    }
  }
}

void MultiLevelQueuePlanner::EnqueueJob(Job job, int queue_level) {
  // There can be other ways to enqueue the job, given the priority of the job.
  multi_level_queue_[queue_level].push_back(job);
}

void MultiLevelQueuePlanner::ScheduleQueue(size_t queue_level,
                                           DeviceWaitingTime& device_waiting) {
  auto& job_queue = multi_level_queue_[queue_level];
  while(!job_queue.empty()) {
    // A simple round-robin logic can be applied here.
    // 1. Find which job to scheduling considering the device waiting time.
    // 2. Enqueue to job to the corresponding worker.
    // 3. update `device_waiting_` status.
    return;
  }
}

bool MultiLevelQueuePlanner::NeedProfile() {
  return true;
}

}  // namespace impl
}  // namespace tflite
