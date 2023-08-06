#include <gtest/gtest.h>
#include <stdarg.h>
#include <stdint.h>

#include <array>
#include <fstream>
#include <vector>

#include "band/backend/grpc/model.h"
#include "band/backend/grpc/model_executor.h"
#include "band/backend/grpc/tensor.h"
#include "band/backend_factory.h"
#include "band/engine.h"
#include "band/interface/model_executor.h"
#include "band/interface/tensor.h"
#include "band/model.h"
#include "band/tensor.h"

namespace band {
namespace grpc {

TEST(GrpcClient, GetModelDesc) {
  auto client = GrpcClient();
  EXPECT_EQ(client.Connect("localhost", 50051), absl::OkStatus());
  client.GetModelDesc();
  
}

}  // namespace grpc
}  // namespace band

int main(int argc, char** argv) {
#ifdef BAND_GRPC
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
#endif  // BAND_GRPC
  return 0;
}
