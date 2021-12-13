#ifndef TENSORFLOW_LITE_PLANNER_UTIL_H_
#define TENSORFLOW_LITE_PLANNER_UTIL_H_

#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/config.h"
#include "tensorflow/lite/safe_bool.h"

namespace tflite {

namespace impl {

// The maximum number of available job outputs at one time.
#define NUM_FINISHED_RECORDS 1000

// Type definition of job queue.
using JobQueue = std::deque<Job>;
// Type definition for the device waiting time.
// The unit of time is ms.
using WorkerWaitingTime = std::map<int, int64_t>;
// Reserved time per worker
// reserved_time[worker_id] = vector of (start time, end time)
using ReservedTime = std::map<int, std::set<std::pair<int64_t, int64_t>>>;
// Decision from a scheduler. The Jobs in the action must be passed to
// the appropriate workers.
using ScheduleAction = std::map<int, std::vector<Job>>;

// The job queue which can be shared by multiple threads.
struct ConcurrentJobQueue {
  JobQueue queue;
  std::mutex mtx;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_UTIL_H_
