#include "band/backend/grpc/backend.h"

namespace band {

bool GrpcRegisterCreators() {
  BackendFactory::RegisterBackendCreators(
      BackendType::Grpc, new grpc::ModelExecutorCreator, new grpc::ModelCreator,
      new grpc::UtilCreator);
  return true;
}

}  // namespace band