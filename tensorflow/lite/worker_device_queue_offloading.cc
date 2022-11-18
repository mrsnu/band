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

#include <iostream>
#include <string>
#include <grpcpp/grpcpp.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>

#include "tensorflow/lite/proto/splash.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using grpc::ClientWriter;
using grpc::ClientReader;
using grpc::WriteOptions;
using splash::Splash;
using splash::Request;
using splash::Response;

class SplashClient {
 public:
  SplashClient(std::shared_ptr<Channel> channel, int data_size)
      : stub_(Splash::NewStub(channel)) {
        dataSize = data_size;
        int height = 112;
        int width = 112;
        int data = height * width * 3;
        std::allocator<char> alloc;
        buffer_ = alloc.allocate(data);
        for (int i = 0; i < data; i++) {
          buffer_[i] = 1;
        }
        buffer_[data - 1] = 0; 
      }

  // Assembles the client's payload, sends it and presents the response back
  // from the server.
  int64_t Invoke(tflite::Subgraph* subgraph) {
    // Data we are sending to the server.
    Request request;
    int height = 112;
    int width = 112;
    int data = height * width * 3;
    for (int i = 0; i < data; i++) {
      buffer_[i] = 1;
    }
    buffer_[data - 1] = 0; 
    request.set_model("model_name");
    request.set_height(height);
    request.set_width(width);
    request.set_data(buffer_);

    Response response;
    ClientContext context;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    Status status = stub_->RequestInference(&context, request, &response);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    LOGI("RPC time : %d", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    if (status.ok()) {
      return response.computation_time_ms();
    } else {
      return -1;
    }
  }

  // std::string UploadFile(const std::string& filename) {
  //   StringResponse response;
  //   ClientContext context;
  //   std::unique_ptr<ClientWriter<UploadFileRequest>> writer(stub_->UploadFile(&context, &response));

  //   UploadFileRequest request;
  //   request.mutable_metadata()->set_filename("1GB");
  //   request.mutable_metadata()->set_extension(".bin");

  //   char buffer[MAX_FILE_SIZE];
  //   LOGI("UploadFile: file transfer %s", filename.c_str());
  //   std::fstream fileStream = std::fstream(filename, std::ios::in | std::ios::binary);
  //   if (!fileStream.is_open()) {
  //     LOGI("File not exists");
  //     return "File not exists"; 
  //   }
  //   std::chrono::steady_clock::time_point total_begin = std::chrono::steady_clock::now();
  //   while (!fileStream.eof()) {
  //     // std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_DELAY_MILLISECONDS));
  //     request.clear_chunk_data();
  //     fileStream.read(buffer, MAX_FILE_SIZE);
  //     request.set_chunk_data(std::string(buffer, MAX_FILE_SIZE));
  //     std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  //     writer->Write(request, WriteOptions().set_buffer_hint());
  //     std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  //     LOGI("UploadFile RPC time : %d ms", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000);
  //   }
  //   std::chrono::steady_clock::time_point total_end = std::chrono::steady_clock::now();
  //   LOGI("UploadFile total time : %d ms", std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_begin).count() / 1000);
  //   writer->WritesDone();
  //   fileStream.close();
  //   LOGI("UploadFile: file transfer %s done", filename.c_str());
  //   Status status = writer->Finish();
  //   if (status.ok()) {
  //     return "RPC success";
  //   } else {
  //     LOGI("RPC failed: %d, : %s", status.error_code(), status.error_message().c_str());
  //     return "RPC failed";
  //   }
  // }

  // std::string DownloadFile(const std::string& filename) {
  //   FileResponse response;
  //   ClientContext context;
  //   MetaData metadata;
  //   metadata.set_filename("1GB");
  //   metadata.set_extension(".bin");

  //   std::unique_ptr<ClientReader<FileResponse>> reader(stub_->DownloadFile(&context, metadata));

  //   LOGI("DownloadFile: file transfer %s", filename.c_str());
  //   std::fstream fileStream = std::fstream(filename, std::ios::out | std::ios::binary);
  //   std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  //   std::chrono::steady_clock::time_point total_begin = std::chrono::steady_clock::now();
  //   while (reader->Read(&response)) {
  //     std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  //     LOGI("DownloadFile RPC time : %d ms", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000);
  //     fileStream.write(response.chunk_data().c_str(), MAX_FILE_SIZE);
  //     response.clear_chunk_data();
  //     begin = std::chrono::steady_clock::now();
  //   }
  //   fileStream.close();
  //   Status status = reader->Finish();
  //   std::chrono::steady_clock::time_point total_end = std::chrono::steady_clock::now();
  //   LOGI("DownloadFile total time : %d ms", std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_begin).count() / 1000);
  //   if (status.ok()) {
  //     return "RPC success";
  //   } else {
  //     LOGI("RPC failed: %d, : %s", status.error_code(), status.error_message().c_str());
  //     return "RPC failed";
  //   }
  // }

 private:
  std::unique_ptr<Splash::Stub> stub_;
  int dataSize;
  char* buffer_;
};

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

bool DeviceQueueOffloadingWorker::IsBusy() {
  return false;
}

int64_t DeviceQueueOffloadingWorker::GetWaitingTime() {
  std::unique_lock<std::mutex> lock(device_mtx_);
  if (!IsAvailable()) {
    return LARGE_WAITING_TIME;
  }
  // assume 100 ms per job (currently, does not check for profiling)
  return 1000000 * requests_.size();
}

int64_t DeviceQueueOffloadingWorker::GetBandwidthMeasurement() {
  std::unique_lock<std::mutex> lock(device_mtx_);
  if (!IsAvailable()) {
    return LARGE_WAITING_TIME;
  }

  std::shared_ptr<Planner> planner = planner_.lock();
  if (!planner) {
    return -1;
  }
  Interpreter* interpreter = planner->GetInterpreter();

  // assume 100 ms per job (currently, does not check for profiling)
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
  SplashClient grpc_client(
  grpc::CreateChannel(offloading_target_, grpc::InsecureChannelCredentials()), offloading_data_size_);
  // random string; copy-pasted from grpc example

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
