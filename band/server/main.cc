#include <grpc/grpc.h>
#include <grpcpp/server_builder.h>

#include "band/server/grpc_server.h"

void RunServer(std::string host, int port) {
  std::string server_address = host + ":" + std::to_string(port);
  band::server::GrpcServerImpl service;
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  server->Wait();
}

int main() {
  RunServer("localhost", 8192);
}
