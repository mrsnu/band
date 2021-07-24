#include "tensorflow/lite/planner/multi_level_queue_planner.h"
#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace impl {

void MultiLevelQueuePlanner::Plan() {
  while (true) {
    if (GetSafeBool().wait())
      return;

    // Move all the jobs to the queue with the highest priority.
    // JobQueue local_jobs;
    CopyToLocalQueue(multi_level_queue_[0]);
    AllocateJobsToQueue(multi_level_queue_[0]);

    // Update the worker status.
    UpdateDeviceWaitingTime();

    // Schedule each queue starting from the queue with the highest priority.
    for (size_t i = 0; i < GetNumQueues(); ++i) {
      if (IsQueueLevelValid(i)) {
        ScheduleQueue(i, device_waiting_);
      } else {
        TFLITE_LOG(WARN) << "The selected queue level is invalid.";
      }
    }
  }
}

void MultiLevelQueuePlanner::EnqueueJob(Job job, int queue_level) {
  // There can be other ways to enqueue the job, given the priority of the job.
  multi_level_queue_[queue_level].push_back(job);
}

void MultiLevelQueuePlanner::AllocateJobsToQueue(JobQueue& local_jobs) {

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
