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
#include "band/config_builder.h"
#include "band/engine.h"
#include "band/interface/model_executor.h"
#include "band/interface/tensor.h"
#include "band/model.h"
#include "band/tensor.h"

namespace band {
using namespace interface;
TEST(GrpcBackend, WriteModel) {
  grpc::GrpcModel bin_model(0);
  bin_model.id = "TestModel";
  bin_model.num_ops = 1;
  bin_model.num_tensors = 1;
  bin_model.tensor_types = {DataType::Float32};
  bin_model.input_tensor_indices = {0};
  bin_model.output_tensor_indices = {0};
  bin_model.op_input_tensors = {{0}};
  bin_model.op_output_tensors = {{0}};
  EXPECT_EQ(bin_model.IsInitialized(), true);
  
  EXPECT_EQ(bin_model.ToPath("test.band"), absl::OkStatus());
}

TEST(GrpcBackend, ReadModel) {
  grpc::GrpcModel bin_model(0);
  EXPECT_EQ(bin_model.FromPath("band/test/data/test.band"), absl::OkStatus());
  EXPECT_EQ(bin_model.IsInitialized(), true);
  EXPECT_EQ(bin_model.id, "TestModel");
  EXPECT_EQ(bin_model.num_ops, 1);
  EXPECT_EQ(bin_model.num_tensors, 1);
  for (int i = 0; i < bin_model.tensor_types.size(); i++) {
    EXPECT_EQ(bin_model.tensor_types[i], DataType::Float32);
  }
  for (int i = 0; i < bin_model.input_tensor_indices.size(); i++) {
    EXPECT_EQ(bin_model.input_tensor_indices[i], i);
  }
  for (int i = 0; i < bin_model.output_tensor_indices.size(); i++) {
    EXPECT_EQ(bin_model.output_tensor_indices[i], i);
  }
  for (int i = 0; i < bin_model.op_input_tensors.size(); i++) {
    for (auto j : bin_model.op_input_tensors[i]) {
      EXPECT_EQ(j, 0);
    }
  }
  for (int i = 0; i < bin_model.op_output_tensors.size(); i++) {
    for (auto j : bin_model.op_output_tensors[i]) {
      EXPECT_EQ(j, 0);
    }
  }
}

}  // namespace band

int main(int argc, char** argv) {
#ifdef BAND_GRPC
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
#endif  // BAND_GRPC
  return 0;
}
