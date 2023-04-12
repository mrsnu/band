#ifndef BAND_BACKEND_GRPC_GRPC_CLIENT_H_
#define BAND_BACKEND_GRPC_GRPC_CLIENT_H_

#include "band/backend/grpc/proto/band.grpc.pb.h"

class GrpcClient {
 public:
  GrpcClient(std::shared_ptr<grpc::Channel> channel)
      : stub_(band::BandService::NewStub(channel)) {}
}

#endif  // BAND_BACKEND_GRPC_GRPC_CLIENT_H_