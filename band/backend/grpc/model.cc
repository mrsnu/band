#include "band/backend/grpc/model.h"

namespace band {
namespace grpc {

GrpcModel::GrpcModel(ModelId id) : interface::IModel(id) {}

BackendType GrpcModel::GetBackendType() const { return BackendType::Grpc; }

absl::Status GrpcModel::FromPath(const char* filename) {
  return absl::OkStatus();
}

absl::Status GrpcModel::FromBuffer(const char* buffer, size_t buffer_size) {
  return absl::OkStatus();
}

bool GrpcModel::IsInitialized() const {
  return false;
}

}  // namespace grpc
}  // namespace band
