#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "band/tensor/image_operation.h"
#include "band/tensor/buffer.h"

namespace band {
namespace test {

TEST(ImageOperationTest, CropOperationTest) {
  tensor::CropOperation crop_op(0, 0, 2, 2);
  std::array<unsigned char, 9> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  tensor::Buffer::CreateFromRaw(input_data.data(), 3, 3, 1, tensor::Buffer::Format::GrayScale, tensor::Buffer::DataType::kUInt8);


  tensor::Buffer input(3, 3, 1);
  input.data<float>()[0] = 1;
  input.data<float>()[1] = 2;
  input.data<float>()[2] = 3;
  input.data<float>()[3] = 4;
  input.data<float>()[4] = 5;
  input.data<float>()[5] = 6;
  input.data<float>()[6] = 7;
  input.data<float>()[7] = 8;
  input.data<float>()[8] = 9;
  tensor::Buffer output(2, 2, 1);
  crop_op.SetOutput(&output);
  crop_op.Process(input);
  EXPECT_EQ(output.data<float>()[0], 1);
  EXPECT_EQ(output.data<float>()[1], 2);
  EXPECT_EQ(output.data<float>()[2], 4);
  EXPECT_EQ(output.data<float>()[3], 5);
}


}  // namespace test
}  // namespace band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}