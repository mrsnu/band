#include "tensorflow/lite/worker.h"

#include <algorithm>

#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/time.h"

#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "libtflite", __VA_ARGS__)
#include <android/log.h>
#endif // defined(__ANDROID__)

namespace tflite {
namespace impl {

bool GlobalQueueOffloadingWorker::GiveJob(Job& job) {
  if (is_busy_ || !IsAvailable()) {
    return false;
  }

  current_job_ = job;
  is_busy_ = true;
  request_cv_.notify_one();
  return true;
}

bool GlobalQueueOffloadingWorker::IsBusy() {
  std::lock_guard<std::mutex> lock(device_mtx_);
  return is_busy_;
}

int GlobalQueueOffloadingWorker::GetCurrentJobId() {
  std::unique_lock<std::mutex> lock(device_mtx_);
  return current_job_.job_id;
}

std::vector<thermal_t> GlobalQueueOffloadingWorker::GetEstimatedEndTemperature() {
  std::unique_lock<std::mutex> lock(device_mtx_);
  // Return dummy values
  std::shared_ptr<Planner> planner_ptr = planner_.lock();
  return planner_ptr->GetResourceMonitor().GetAllTemperature();
}

int64_t GlobalQueueOffloadingWorker::GetWaitingTime() {
  std::unique_lock<std::mutex> lock(device_mtx_);
  if (!IsAvailable()) {
    return LARGE_WAITING_TIME;
  }

  if (!is_busy_) {
    return 0;
  }
  return 0;
}

void GlobalQueueOffloadingWorker::Work() {
  while (true) {
    std::unique_lock<std::mutex> lock(device_mtx_);
    request_cv_.wait(lock, [this]() {
      return (kill_worker_ || is_busy_) && !is_paused_;
    });

    if (kill_worker_) {
      break;
    }

    lock.unlock();

    if (!IsValid(current_job_)) {
      TF_LITE_MAYBE_REPORT_ERROR(
          GetErrorReporter(),
          "%s worker failed to acquire ptr to planner",
          TfLiteDeviceGetName(device_flag_));
      break;
    }

    int subgraph_idx = current_job_.subgraph_idx;
    std::shared_ptr<Planner> planner_ptr = planner_.lock();
    if (planner_ptr) {
      Interpreter* interpreter_ptr = planner_ptr->GetInterpreter();
      lock.lock();
      current_job_.invoke_time = profiling::time::NowMicros();
      planner_ptr->GetResourceMonitor().FillJobInfoBefore(current_job_);
      lock.unlock();

      // TODO: Need to send and receive input/output tensors
      // std::string reply = greeter.SayHello(user);
      // std::string reply = greeter.UploadFile("/data/data/com.aaa.cj.offloading/1GB.bin");
      // std::string reply2 = greeter.DownloadFile("/data/data/com.aaa.cj.offloading/1GB_download.bin");
      // std::cout << "Greeter received: " << reply << std::endl;
      switch (current_job_.model_id) {
        case 0://"retinaface_mbv2_quant_160.tflite":
          std::this_thread::sleep_for(std::chrono::milliseconds(101));
          break;
        case 1://"arc_mbv2_quant.tflite":
          std::this_thread::sleep_for(std::chrono::milliseconds(27));
          break;
        case 2://"arc_res50_quant.tflite":
          std::this_thread::sleep_for(std::chrono::milliseconds(103));
          break;
        case 3://"ICN_quant.tflite":
          std::this_thread::sleep_for(std::chrono::milliseconds(45));
          break;
        default:
          std::this_thread::sleep_for(std::chrono::milliseconds(40));
          break;
      }

      planner_ptr->GetResourceMonitor().FillJobInfoAfter(current_job_);
      current_job_.end_time = profiling::time::NowMicros();
      current_job_.latency = current_job_.end_time - current_job_.invoke_time;

      interpreter_ptr->UpdateExpectedLatency(subgraph_idx, current_job_.latency);

      planner_ptr->GetModelManager()->Update(current_job_);
      if (current_job_.following_jobs.size() != 0) {
        planner_ptr->EnqueueBatch(current_job_.following_jobs);
      }
      planner_ptr->EnqueueFinishedJob(current_job_);

      lock.lock();
      is_busy_ = false;
      lock.unlock();

      planner_ptr->GetSafeBool().notify();
    } else {
      // TODO #21: Handle errors in multi-thread environment
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
