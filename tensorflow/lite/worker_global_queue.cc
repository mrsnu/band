#include "tensorflow/lite/worker.h"

#include <algorithm>

#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace impl {

bool GlobalQueueWorker::GiveJob(Job& job) {
  std::lock_guard<std::mutex> lock(device_mtx_);
  if (is_busy_) {
    // I'm busy so I can't receive your request
    return false;
  }

  current_job_ = job;
  is_busy_ = true;
  request_cv_.notify_one();
  return true;
}

bool GlobalQueueWorker::IsBusy() {
  std::lock_guard<std::mutex> lock(device_mtx_);
  return is_busy_;
}

// This function returns the remaining time until this worker can start
// processing another Job.
//
// The remaining time is calculated based on the profiled model time of the
// Job, the timestamp of when this worker started processing the Job
// (current_job_.invoke_time), and the current timestamp.
// In case more time has passed (since invoke_time) than the profiled model
// time, this function returns 0, as it is unable to predict when the current
// job will finish.
// This function can also return 0 if the worker is not working on any job at
// the moment (IsBusy() returns false).
//
// In case this function fails to acquire a shared ptr to the Planner,
// we print an error message and this function returns -1.
int64_t GlobalQueueWorker::GetWaitingTime() {
  std::unique_lock<std::mutex> lock(device_mtx_);
  if (!is_busy_) {
    return 0;
  }

  int64_t invoke_time = current_job_.invoke_time;

  // if this thread is the same thread that updates is_busy_ (false --> true)
  // and there are no other threads that call this function, then it is
  // technically safe to unlock here because the worker thread does not
  // update the other fields of current_job_
  // consider unlocking here if we need that teensy little extra perf boost
  // lock.unlock();

  int model_id = current_job_.model_id;
  int start_idx = current_job_.start_idx;
  int end_idx = current_job_.end_idx;

  // we no longer read from this worker's member variables, so there is
  // no need to hold on to the lock anymore
  lock.unlock();

  std::shared_ptr<Planner> planner = planner_.lock();
  if (!planner) {
    TFLITE_LOG(ERROR) << "Worker " << device_flag_
                      << " failed to acquire ptr to planner";
    return -1;
  }
  Interpreter* interpreter = planner->GetInterpreter();

  // TODO #80: Get profiled_latency from current_job_
  SubgraphKey key(model_id, device_flag_, start_idx, end_idx);
  int64_t profiled_latency = interpreter->GetSubgraphProfileResult(key);

  if (invoke_time == 0) {
    // the worker has not started on processing the job yet
    return profiled_latency;
  }

  int64_t current_time = profiling::time::NowMicros();
  int64_t progress = current_time - invoke_time;
  return std::max((long) (profiled_latency - progress), 0L);
}

void GlobalQueueWorker::Work() {
  while (true) {
    std::unique_lock<std::mutex> lock(device_mtx_);
    request_cv_.wait(lock, [this]() {
      return kill_worker_ || is_busy_;
    });

    if (kill_worker_) {
      break;
    }

    lock.unlock();

    if (current_job_.model_id < 0 ||
        current_job_.subgraph_idx < 0 ||
        current_job_.device_id < 0 ||
        current_job_.enqueue_time <= 0 ||
        current_job_.invoke_time != 0 ||
        current_job_.end_time != 0) {
      TFLITE_LOG(ERROR) << "Worker " << device_flag_
                        << " spotted an invalid job";
      break;
    }

    int subgraph_idx = current_job_.subgraph_idx;
    std::shared_ptr<Planner> planner_ptr = planner_.lock();
    if (planner_ptr) {
      Interpreter* interpreter_ptr = planner_ptr->GetInterpreter();
      Subgraph& subgraph = *(interpreter_ptr->subgraph(subgraph_idx));

      {
        std::lock_guard<std::mutex> cpu_lock(cpu_set_mtx_);
        if (need_cpu_set_update_) {
          need_cpu_set_update_ = false;

          auto internal_backend = interpreter_ptr->GetCpuBackendContext()
                                      ->internal_backend_context();
          internal_backend->SetCpuSet(std::this_thread::get_id(), cpu_set_);

          if (SetCPUThreadAffinity(cpu_set_) != kTfLiteOk) {
            // TODO #21: Handle errors in multi-thread environment
            TFLITE_LOG(ERROR) << "Worker " << device_flag_
                              << " failed to set cpu thread affinity";
            break;
          }
        }
      }

      lock.lock();
      current_job_.invoke_time = profiling::time::NowMicros();
      lock.unlock();

      if (subgraph.Invoke() == kTfLiteOk) {
        // end_time is never read/written by any other thread as long as
        // is_busy == true, so it's safe to update it w/o grabbing the lock
        current_job_.end_time = profiling::time::NowMicros();
        interpreter_ptr->UpdateProfileResult(
            subgraph.GetKey(),
            (current_job_.end_time - current_job_.invoke_time));
        // TODO #65: Tensor communications between subgraphs
        interpreter_ptr->InvokeModelsAsync(current_job_.following_jobs);
        planner_ptr->EnqueueFinishedJob(current_job_);

      } else {
        // end_time is never read/written by any other thread as long as
        // is_busy == true, so it's safe to update it w/o grabbing the lock
        current_job_.end_time = profiling::time::NowMicros();
        // TODO #21: Handle errors in multi-thread environment
        // Currently, put a job with a minus sign if Invoke() fails.
        // Model 0 fail --> Job(-1), Model 1 fail --> Job(-2), ...
        planner_ptr->EnqueueFinishedJob(Job(-1 * current_job_.model_id - 1));
      }

      lock.lock();
      is_busy_ = false;
      lock.unlock();

      planner_ptr->GetSafeBool().notify();
    } else {
      // TODO #21: Handle errors in multi-thread environment
      TFLITE_LOG(ERROR) << "Worker " << device_flag_
                        << " failed to acquire ptr to planner";
      return;
    }
  }
}

}  // namespace impl
}  // namespace tflite
