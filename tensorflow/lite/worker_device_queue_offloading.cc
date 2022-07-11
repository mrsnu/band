// Temporal usage for debugging
#include <android/log.h>
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "libtflite", __VA_ARGS__)

#include "tensorflow/lite/worker.h"

#include <algorithm>

#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/time.h"

#include <iostream>
#include <string>
#include <grpcpp/grpcpp.h>

#include "tensorflow/lite/proto/helloworld.grpc.pb.h"


using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using helloworld::Greeter;
using helloworld::HelloReply;
using helloworld::HelloRequest;


/*
 *******************************************************************
 * Copy-pasted from grpc/examples/cpp/helloworld/greeter_client.cc *
 *******************************************************************
 */
class GreeterClient {
 public:
  GreeterClient(std::shared_ptr<Channel> channel)
      : stub_(Greeter::NewStub(channel)) {}

  // Assembles the client's payload, sends it and presents the response back
  // from the server.
  std::string SayHello(const std::string& user) {
    // Data we are sending to the server.
    HelloRequest request;
    request.set_name(user);

    // Container for the data we expect from the server.
    HelloReply reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // The actual RPC.
    Status status = stub_->SayHello(&context, request, &reply);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    LOGI("RPC time : %d", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000 );

    // Act upon its status.
    if (status.ok()) {
      LOGI("RPC OK : %s", reply.message().c_str());
      return reply.message();
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
      LOGI("RPC failed: %d, : %s", status.error_code(), status.error_message().c_str());
      return "RPC failed";
    }
  }

 private:
  std::unique_ptr<Greeter::Stub> stub_;
};

namespace tflite {
namespace impl {

JobQueue& DeviceQueueOffloadingWorker::GetDeviceRequests() {
  return requests_;
}

void DeviceQueueOffloadingWorker::AllowWorkSteal() {
  allow_work_steal_ = true;
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
  GreeterClient greeter(
  grpc::CreateChannel(offloading_target_, grpc::InsecureChannelCredentials()));
  // random string; copy-pasted from grpc example
  std::string user("world");

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

    std::shared_ptr<Planner> planner_ptr = planner_.lock();
    if (planner_ptr) {
      if (TryUpdateWorkerThread() != kTfLiteOk) {
        // TODO #21: Handle errors in multi-thread environment
        break;
      }

      lock.lock();
      current_job.invoke_time = profiling::time::NowMicros();
      lock.unlock();

      // TODO: Need to send and receive input/output tensors
      std::string reply = greeter.SayHello(user);
      // std::cout << "Greeter received: " << reply << std::endl;

      current_job.end_time = profiling::time::NowMicros();
      current_job.status = kTfLiteJobSuccess;

      if (current_job.following_jobs.size() != 0) {
        planner_ptr->EnqueueBatch(current_job.following_jobs);
      }

      planner_ptr->EnqueueFinishedJob(current_job);

      lock.lock();
      requests_.pop_front();
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

void DeviceQueueOffloadingWorker::TryWorkSteal() {
  // not supported
}

}  // namespace impl
}  // namespace tflite
