#ifndef TENSORFLOW_LITE_SPLASH_GRPC_CLIENT_H_
#define TENSORFLOW_LITE_SPLASH_GRPC_CLIENT_H_

#include <vector>
#include <string>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/config.h"

#include <grpcpp/grpcpp.h>

#include "tensorflow/lite/proto/splash.grpc.pb.h"

#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "libtflite", __VA_ARGS__)
#include <android/log.h>
#else
#define LOGI(...) printf(__VA_ARGS__)
#endif // defined(__ANDROID__)

using grpc::Channel;
using grpc::ClientContext;
using grpc::CompletionQueue;
using grpc::Status;
using grpc::ClientWriter;
using grpc::ClientReader;
using grpc::WriteOptions;
using grpc::ClientAsyncResponseReader;
using splash::Splash;
using splash::Request;
using splash::Response;

class SplashGrpcClient {
 public:
  SplashGrpcClient(std::shared_ptr<Channel> channel, int data_size)
      : stub_(Splash::NewStub(channel)) {
        dataSize = data_size;
        int height = 130;
        int width = 130;
        int data = height * width * 3;
        std::allocator<char> alloc;
        buffer_1_ = alloc.allocate(data);
        dummy_1_ = alloc.allocate(data);
        for (int i = 0 ; i < data; i++) {
          dummy_1_[i] = 1;
        }
        dummy_1_[data] = 0;
        memcpy(buffer_1_, dummy_1_, data);

        int height2 = 112;
        int width2 = 112;
        int data2 = height * width * 3;
        buffer_2_ = alloc.allocate(data2);
        dummy_2_ = alloc.allocate(data2);
        for (int i = 0 ; i < data2; i++) {
          dummy_2_[i] = 1;
        }
        dummy_2_[data2] = 0;
        memcpy(buffer_2_, dummy_2_, data2);
  }

  int64_t Invoke(tflite::Subgraph* subgraph) {
    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    Request request;
    const std::vector<int>& input_tensors = subgraph->inputs();
    const int tensor_idx = input_tensors[0]; // Now, only supporting for one input model
    int64_t data_size = subgraph->tensor(tensor_idx)->bytes;
    // if (subgraph->GetKey().model_id == 0) {
    //   request.set_model("face_detection");
    //   request.set_height(160);
    //   request.set_width(160);
    //   request.set_data(buffer_1_);
    // } else if (subgraph->GetKey().model_id == 1) {
    //   request.set_model("face_recognition");
    //   request.set_height(112);
    //   request.set_width(112);
    //   request.set_data(buffer_2_);
    // }
    if (subgraph->GetKey().model_id == 0) {
      request.set_model("object_detection");
      request.set_height(320);
      request.set_width(320);
      request.set_data(buffer_1_);
    // } else if (subgraph->GetKey().model_id == 1) {
    //   request.set_model("text_detection");
    //   request.set_height(320);
    //   request.set_width(320);
    //   request.set_data(buffer_2_);
    } else if (subgraph->GetKey().model_id == 1) {
      request.set_model("image_classification");
      request.set_height(224);
      request.set_width(224);
      request.set_data(buffer_3_);
    }

    Response response;
    ClientContext context;

    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // LOGI("copy time : %d", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    Status status = stub_->RequestInference(&context, request, &response);
    if (status.ok()) {
      return response.computation_time_ms();
    } else {
      return -1;
    }
    
    // CompletionQueue cq;

    // std::unique_ptr<ClientAsyncResponseReader<Response>> rpc(stub_->AsyncRequestInference(&context, request, &cq));
    // // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // // LOGI("copy time : %d", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    // Status status;
    // rpc->Finish(&response, &status, (void*)1);
    // void* got_tag;
    // bool ok = false;
    // cq.Next(&got_tag, &ok);
    // if (ok && got_tag == (void*)1) {
    //   return response.computation_time_ms();
    // } else {
    //   return -1;
    // }
  }

  int64_t Ping() {
    Request request;
    request.set_model("nothing");
    request.set_height(130);
    request.set_width(130);
    request.set_data(buffer_1_);

    Response response;
    ClientContext context;

    Status status = stub_->RequestInference(&context, request, &response);
    if (status.ok()) {
      return response.computation_time_ms();
    } else {
      return -1;
    }
  }

  // std::string InvokeStream(tflite::Subgraph* subgraph) {
  //   const std::vector<int>& input_tensors = subgraph->inputs();
  //   const int tensor_idx = input_tensors[0]; // Now, only supporting for one input model
  //   int64_t data_size = subgraph->tensor(tensor_idx)->bytes;

  //   if (subgraph->GetKey().model_id == 0) {
  //     request.set_model("object_detection");
  //     request.set_height(320);
  //     request.set_width(320);
  //     request.set_data(buffer_1_);
  //   // } else if (subgraph->GetKey().model_id == 1) {
  //   //   request.set_model("text_detection");
  //   //   request.set_height(320);
  //   //   request.set_width(320);
  //   //   request.set_data(buffer_2_);
  //   } else if (subgraph->GetKey().model_id == 1) {
  //     request.set_model("image_classification");
  //     request.set_height(224);
  //     request.set_width(224);
  //     request.set_data(buffer_3_);
  //   }

  //   StringResponse response;
  //   ClientContext context;
  //   std::unique_ptr<ClientWriter<UploadFileRequest>> writer(stub_->UploadFile(&context, &response));

  //   UploadFileRequest request;

  //   int64_t remain_bytes = data_size;
  //   while (remain_bytes > 0) {
  //     request.clear_chunk_data();
  //     request.set_chunk_data(std::string(buffer_1_, MAX_FILE_SIZE));
  //     writer->Write(request, WriteOptions().set_buffer_hint());
  //     remain_bytes -= MAX_FILE_SIZE;
  //   }
  //   writer->WritesDone();
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
  char* buffer_1_;
  char* buffer_2_;
  char* buffer_3_;
  char* dummy_1_;
  char* dummy_2_;
  char* dummy_3_;
};

#endif // TENSORFLOW_LITE_SPLASH_GRPC_CLIENT_H_