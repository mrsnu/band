#ifndef BAND_BACKEND_GRPC_GRPC_CLIENT_H_
#define BAND_BACKEND_GRPC_GRPC_CLIENT_H_

#include <grpcpp/grpcpp.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "band/backend/grpc/proto/service.grpc.pb.h"
#include "band/model_spec.h"

namespace {

std::shared_ptr<grpc::Channel> GetChannel(std::string host, int port) {
  return grpc::CreateChannel(host + ":" + std::to_string(port),
                             grpc::InsecureChannelCredentials());
}

}  // namespace

class GrpcClientService {
 public:
  GrpcClientService(std::shared_ptr<grpc::Channel> channel)
      : stub_(band_proto::BandService::NewStub(channel)) {}

  absl::StatusOr<std::vector<band_proto::ModelDescriptor>> GetModelDesc() const;
  absl::StatusOr<band_proto::Status> CheckModelDesc(
      band_proto::ModelDescriptor model_desc) const;
  absl::StatusOr<band_proto::Response> RequestSync(
      band_proto::Request request) const;

 private:
  std::unique_ptr<band_proto::BandService::Stub> stub_;
};

namespace band {
namespace grpc {

class GrpcClient {
 public:
  GrpcClient() = default;
  void Connect(std::string host, int port) {
    client_ = std::make_unique<GrpcClientService>(GetChannel(host, port));
  }

  absl::StatusOr<std::vector<band_proto::ModelDescriptor>> GetModelDesc() const;
  absl::Status CheckModelDesc(band_proto::ModelDescriptor model_desc) const;
  absl::StatusOr<band_proto::Response> RequestSync(band_proto::Request) const;

 private:
  std::unique_ptr<GrpcClientService> client_;
};

}  // namespace grpc
}  // namespace band

#endif  // BAND_BACKEND_GRPC_GRPC_CLIENT_H_