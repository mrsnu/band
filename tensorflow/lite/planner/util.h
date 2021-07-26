#ifndef TENSORFLOW_LITE_PLANNER_UTIL_H_
#define TENSORFLOW_LITE_PLANNER_UTIL_H_

#include <memory>
#include <vector>
#include <string>
#include <deque>
#include <mutex>

#include "tensorflow/lite/worker.h"
#include "tensorflow/lite/safe_bool.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/config.h"

namespace tflite {

namespace impl {

// The maximum number of available job outputs at one time.
#define NUM_FINISHED_RECORDS 1000

typedef enum TfLitePlannerType {
  kFixedDevice = 0,
  kRoundRobin = 1,
  kShortestExpectedLatency = 2,
  kFixedDeviceGlobalQueue = 3,
  kNumPlannerTypes = 4
} TfLitePlannerType;

// Type definition of job queue.
using JobQueue = std::deque<Job>;
// Type definition for the device waiting time.
using DeviceWaitingTime = std::map<TfLiteDeviceFlags, int64_t>;
// Decision from a scheduler. The Jobs in the action must be passed to
// the appropriate workers.
using ScheduleAction = std::map<TfLiteDeviceFlags, std::vector<Job>>;

// The job queue which can be shared by multiple threads.
struct ConcurrentJobQueue {
  JobQueue queue;
  std::mutex mtx;
};

class Scheduler {
 public:
  bool NeedProfile() {
    return need_profile_;
  }

  WorkerType GetWorkerType() {
    return worker_type_;
  }

  ScheduleAction Schedule() {
    return schedule();
  }

 protected:
  std::function<ScheduleAction()> schedule;
  bool need_profile_;
  WorkerType worker_type_;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_UTIL_H_
