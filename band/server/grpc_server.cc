#include "band/server/grpc_server.h"

namespace band {
namespace server {

::grpc::Status GrpcServerImpl::GetModelDesc(
    ::grpc::ServerContext* context, const ::band_proto::Void* request,
    ::grpc::ServerWriter< ::band_proto::ModelDescriptor>* writer) {
  return ::grpc::Status::OK;
}
::grpc::Status GrpcServerImpl::CheckModelDesc(
    ::grpc::ServerContext* context,
    const ::band_proto::ModelDescriptor* request,
    ::band_proto::Status* response) {
  return ::grpc::Status::OK;
}
::grpc::Status GrpcServerImpl::RequestSync(::grpc::ServerContext* context,
                                           const ::band_proto::Request* request,
                                           ::band_proto::Response* response) {
  return ::grpc::Status::OK;
}

}  // namespace server
}  // namespace band