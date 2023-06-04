#include "band/backend/grpc/grpc_client.h"

absl::StatusOr<std::vector<band_proto::ModelDescriptor>>
GrpcClientService::GetModelDesc() const {
  std::vector<band_proto::ModelDescriptor> model_descs;
  grpc::ClientContext context;
  auto reader = stub_->GetModelDesc(&context, band_proto::Void());
  band_proto::ModelDescriptor model_desc;
  while (reader->Read(&model_desc)) {
    model_descs.push_back(model_desc);
  }
  auto status = reader->Finish();
  if (!status.ok()) {
    return absl::InternalError(status.error_message());
  }
  return model_descs;
}

absl::StatusOr<band_proto::Status> GrpcClientService::CheckModelDesc(
    band_proto::ModelDescriptor model_desc) const {
  grpc::ClientContext context;
  band_proto::Status status;
  auto req_status = stub_->CheckModelDesc(&context, model_desc, &status);
  if (!req_status.ok()) {
    return absl::InternalError(req_status.error_message());
  }
  return status;
}

absl::StatusOr<band_proto::Response> GrpcClientService::RequestSync(
    band_proto::Request request) const {
  grpc::ClientContext context;
  band_proto::Response response;
  auto req_status = stub_->RequestSync(&context, request, &response);
  if (!req_status.ok()) {
    return absl::InternalError(req_status.error_message());
  }
  return response;
}

namespace band {
namespace grpc {

absl::StatusOr<std::vector<band_proto::ModelDescriptor>>
GrpcClient::GetModelDesc() const {
  auto status_or_descs = client_.GetModelDesc();
  if (!status_or_descs.ok()) {
    return status_or_descs.status();
  }
  return status_or_descs.value();
}

absl::Status GrpcClient::CheckModelDesc(
    band_proto::ModelDescriptor model_desc) const {
  auto status_or_res = client_.CheckModelDesc(model_desc);
  if (!status_or_res.ok()) {
    return status_or_res.status();
  }
  auto res = status_or_res.value();
  if (res.code() != band_proto::StatusCode::OK) {
    return absl::InternalError(res.error_message());
  }
  return absl::OkStatus();
}

absl::StatusOr<band_proto::Response> GrpcClient::RequestSync(
    band_proto::Request) const {
  return band_proto::Response();
}

}  // namespace grpc
}  // namespace band