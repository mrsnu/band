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

// Type definition of job queue.
using JobQueue = std::deque<Job>;

// The job queue which can be shared by multiple threads.
struct ConcurrentJobQueue {
  JobQueue queue;
  std::mutex mtx;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_UTIL_H_
