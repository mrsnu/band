#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "libtflite", __VA_ARGS__)
#include <android/log.h>
#endif // defined(__ANDROID__)

#define MAX_FILE_SIZE 102400
#define SLEEP_DELAY_MILLISECONDS 100 

#include "tensorflow/lite/worker.h"

#include <algorithm>

#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/splash/splash_grpc_client.h"

#include <iostream>
#include <string>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>

namespace tflite {
namespace impl {

JobQueue& DeviceQueueOffloadingWorker::GetDeviceRequests() {
  return requests_;
}

int DeviceQueueOffloadingWorker::GetCurrentJobId() {
  std::unique_lock<std::mutex> lock(device_mtx_);
  if (requests_.empty()) {
    return -1;
  }
  return requests_.front().job_id;
}

int64_t DeviceQueueOffloadingWorker::GetWaitingTime() {
  std::unique_lock<std::mutex> lock(device_mtx_);
  if (!IsAvailable()) {
    return LARGE_WAITING_TIME;
  }
  // TODO: not implemented yet 
  return 1000000 * requests_.size();
}

bool DeviceQueueOffloadingWorker::GiveJob(Job& job) {
  if (!IsAvailable()) {
    return false;
  }

  requests_.push_back(job);
  request_cv_.notify_one();
  return true;
}

void DeviceQueueOffloadingWorker::Work() {
  SplashGrpcClient grpc_client(
  grpc::CreateChannel(offloading_target_, grpc::InsecureChannelCredentials()), offloading_data_size_);
  LOGI("Offloading target: %s", offloading_target_.c_str());
  while (true) {
    std::unique_lock<std::mutex> lock(device_mtx_);
    request_cv_.wait(lock, [this]() {
      return (kill_worker_ || !requests_.empty()) && !is_paused_;
    });

    if (kill_worker_) {
      break;
    }

    Job& current_job = requests_.front();
    lock.unlock();

    if (!IsValid(current_job)) {
      TF_LITE_MAYBE_REPORT_ERROR(
          GetErrorReporter(),
          "%s worker spotted an invalid job",
          TfLiteDeviceGetName(device_flag_));
      break;
    }

    int subgraph_idx = current_job.subgraph_idx;
    std::shared_ptr<Planner> planner_ptr = planner_.lock();
    if (planner_ptr) {
      Interpreter* interpreter_ptr = planner_ptr->GetInterpreter();
      Subgraph* subgraph = interpreter_ptr->subgraph(subgraph_idx);

      lock.lock();
      current_job.invoke_time = profiling::time::NowMicros();
      planner_ptr->GetResourceMonitor().FillJobInfoBefore(current_job);
      lock.unlock();

      int64_t computation_time = grpc_client.Invoke(subgraph);

      planner_ptr->GetResourceMonitor().FillJobInfoAfter(current_job);
      current_job.end_time = profiling::time::NowMicros();
      current_job.latency = current_job.end_time - current_job.invoke_time;
      current_job.communication_time = current_job.latency - computation_time;
      current_job.status = kTfLiteJobSuccess;

      planner_ptr->GetModelManager()->Update(current_job, subgraph);

      if (current_job.following_jobs.size() != 0) {
        planner_ptr->EnqueueBatch(current_job.following_jobs);
      }
      planner_ptr->EnqueueFinishedJob(current_job);

      lock.lock();
      requests_.pop_front();
      lock.unlock();

      planner_ptr->GetSafeBool().notify();
    } else {
      TF_LITE_MAYBE_REPORT_ERROR(
          GetErrorReporter(),
          "%s worker failed to acquire ptr to planner",
          TfLiteDeviceGetName(device_flag_));
      return;
    }
  }
}

}  // namespace impl
}  // namespace tflite
