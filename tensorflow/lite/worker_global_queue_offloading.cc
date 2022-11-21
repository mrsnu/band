#include "tensorflow/lite/worker.h"

#include <algorithm>

#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/time.h"

#include <grpcpp/grpcpp.h>

#include "tensorflow/lite/proto/splash.grpc.pb.h"

#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "libtflite", __VA_ARGS__)
#include <android/log.h>
#endif // defined(__ANDROID__)

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using grpc::ClientWriter;
using grpc::ClientReader;
using grpc::WriteOptions;
using splash::Splash;
using splash::Request;
using splash::Response;

class SplashGlobalClient {
 public:
  SplashGlobalClient(std::shared_ptr<Channel> channel, int data_size)
      : stub_(Splash::NewStub(channel)) {
        dataSize = data_size;
        int height = 160;
        int width = 160;
        int data = height * width * 3;
        std::allocator<char> alloc;
        buffer_1_ = alloc.allocate(data);

        int height2 = 112;
        int width2 = 112;
        int data2 = height * width * 3;
        buffer_2_ = alloc.allocate(data2);
      }

  int64_t Invoke(tflite::Subgraph* subgraph) {
    Request request;
    // FIXME : input to buffer
    int input_size = EstimateInputSize(subgraph);
    if (subgraph->GetKey().model_id == 0) {
      for (int i = 0; i < input_size; i++) {
        buffer_1_[i] = 1;
      }
      buffer_1_[input_size - 1] = 0; 
      request.set_model("face_detection");
      request.set_height(160);
      request.set_width(160);
      request.set_data(buffer_1_);
    } else if (subgraph->GetKey().model_id == 1) {
      for (int i = 0; i < input_size; i++) {
        buffer_2_[i] = 1;
      }
      buffer_2_[input_size - 1] = 0; 
      request.set_model("face_recognition");
      request.set_height(112);
      request.set_width(112);
      request.set_data(buffer_2_);
    }

    Response response;
    ClientContext context;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    Status status = stub_->RequestInference(&context, request, &response);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // LOGI("RPC time : %d", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    if (status.ok()) {
      return response.computation_time_ms();
    } else {
      return -1;
    }
  }

 private:
  std::unique_ptr<Splash::Stub> stub_;
  int dataSize;
  char* buffer_1_;
  char* buffer_2_;

  int64_t EstimateInputSize(const tflite::Subgraph* subgraph) {
    const std::vector<int>& input_tensors = subgraph->inputs();
    int64_t subgraph_input_size = 0;
    for (int tensor_idx : input_tensors) {
      subgraph_input_size += (int64_t)subgraph->tensor(tensor_idx)->bytes;
    }
    return subgraph_input_size;
  }
};

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

int64_t GlobalQueueOffloadingWorker::GetWaitingTime() {
  std::unique_lock<std::mutex> lock(device_mtx_);
  if (!IsAvailable()) {
    return LARGE_WAITING_TIME;
  }

  if (!is_busy_) {
    return 0;
  }

  int64_t invoke_time = current_job_.invoke_time;
  lock.unlock();

  std::shared_ptr<Planner> planner = planner_.lock();
  if (!planner) {
    TF_LITE_MAYBE_REPORT_ERROR(
        GetErrorReporter(),
        "%s worker failed to acquire ptr to planner",
        TfLiteDeviceGetName(device_flag_));
    return -1;
  }
  Interpreter* interpreter = planner->GetInterpreter();

  Subgraph* current_subgraph = interpreter->subgraph(current_job_.subgraph_idx);
  int64_t profiled_latency =
      planner->GetModelManager()->GetPredictedLatency(current_job_.worker_id, current_subgraph);

  if (invoke_time == 0) {
    return profiled_latency;
  }

  int64_t current_time = profiling::time::NowMicros();
  int64_t progress = current_time - invoke_time;
  return std::max((long) (profiled_latency - progress), 0L);
}

void GlobalQueueOffloadingWorker::Work() {
  SplashGlobalClient grpc_client(
  grpc::CreateChannel(offloading_target_, grpc::InsecureChannelCredentials()), offloading_data_size_);
  LOGI("Offloading target: %s", offloading_target_.c_str());
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
      Subgraph* subgraph = interpreter_ptr->subgraph(subgraph_idx);
      lock.lock();
      current_job_.invoke_time = profiling::time::NowMicros();
      planner_ptr->GetResourceMonitor().FillJobInfoBefore(current_job_);
      lock.unlock();

      // std::this_thread::sleep_for(std::chrono::milliseconds(5));
      // int64_t computation_time = 2000;
      int64_t computation_time = grpc_client.Invoke(subgraph);

      planner_ptr->GetResourceMonitor().FillJobInfoAfter(current_job_);
      current_job_.end_time = profiling::time::NowMicros();
      current_job_.latency = current_job_.end_time - current_job_.invoke_time;
      current_job_.communication_time = current_job_.latency - computation_time;

      interpreter_ptr->UpdateExpectedLatency(subgraph_idx, current_job_.latency);

      planner_ptr->GetModelManager()->Update(current_job_, subgraph);
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
