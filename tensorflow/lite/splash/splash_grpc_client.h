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

class SplashGrpcClient {
 public:
  SplashGrpcClient(std::shared_ptr<Channel> channel, int data_size)
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

#endif // TENSORFLOW_LITE_SPLASH_GRPC_CLIENT_H_