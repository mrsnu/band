#ifndef BAND_SERVER_GRPC_SERVER_H_
#define BAND_SERVER_GRPC_SERVER_H_

#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>

#include "band/backend/grpc/proto/service.grpc.pb.h"

namespace band {
namespace server {

class GrpcServerImpl final : public band_proto::BandService::Service {
 public:
  ::grpc::Status GetModelDesc(
      ::grpc::ServerContext* context, const ::band_proto::Void* request,
      ::grpc::ServerWriter< ::band_proto::ModelDescriptor>* writer) override;
  ::grpc::Status CheckModelDesc(::grpc::ServerContext* context,
                                const ::band_proto::ModelDescriptor* request,
                                ::band_proto::Status* response) override;
  ::grpc::Status RequestSync(::grpc::ServerContext* context,
                             const ::band_proto::Request* request,
                             ::band_proto::Response* response) override;
};

}  // namespace server
}  // namespace band

#endif  // BAND_SERVER_GRPC_SERVER_H_