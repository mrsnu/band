#include "band/buffer/image_operation.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "band/buffer/buffer.h"
#include "band/test/image_util.h"

namespace band {
namespace test {

TEST(ImageOperationTest, CropOperationSimpleTest) {
  CropOperation crop_op(0, 0, 1, 1);
  std::array<unsigned char, 9> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto input_buffer =
      Buffer::CreateFromRaw(input_data.data(), 3, 3, BufferFormat::GrayScale);
  EXPECT_EQ(crop_op.Process(*input_buffer), absl::OkStatus());
  auto output_buffer = crop_op.GetOutput();
  EXPECT_EQ(output_buffer->GetDimension()[0], 2);
  EXPECT_EQ(output_buffer->GetDimension()[1], 2);
  EXPECT_EQ(output_buffer->GetNumPlanes(), 1);
  EXPECT_EQ(output_buffer->GetBufferFormat(), BufferFormat::GrayScale);
  EXPECT_EQ(output_buffer->GetOrientation(), BufferOrientation::TopLeft);
  EXPECT_EQ(output_buffer->GetNumElements(), 4);
  EXPECT_EQ((*output_buffer)[0].data[0], 1);
  EXPECT_EQ((*output_buffer)[0].data[1], 2);
  EXPECT_EQ((*output_buffer)[0].data[2], 4);
  EXPECT_EQ((*output_buffer)[0].data[3], 5);
}

TEST(ImageOperationTest, CropOperationImageTest) {
  CropOperation crop_op(0, 0, 255, 255);
  std::shared_ptr<Buffer> input_buffer = LoadImage("band/test/data/hippo.jpg");
  std::shared_ptr<Buffer> cropped_buffer =
      LoadImage("band/test/data/hippo_crop_256.jpg");
  EXPECT_EQ(crop_op.Process(*input_buffer), absl::OkStatus());
  auto output_buffer = crop_op.GetOutput();
  EXPECT_EQ(output_buffer->GetDimension()[0], 256);
  EXPECT_EQ(output_buffer->GetDimension()[1], 256);

  for (size_t i = 0; i < output_buffer->GetNumElements(); ++i) {
    EXPECT_EQ((*output_buffer)[0].data[i], (*cropped_buffer)[0].data[i]);
  }
}

TEST(ImageOperationTest, ConvertImageTest) {
  ConvertOperation convert_op;
  // load 3-channel images
  std::shared_ptr<Buffer> rgb_buffer = LoadImage("band/test/data/hippo.jpg");

  EXPECT_EQ(rgb_buffer->GetBufferFormat(), BufferFormat::RGB);
  // convert to gray scale
  std::shared_ptr<Buffer> output_buffer = Buffer::CreateEmpty(
      rgb_buffer->GetDimension()[0], rgb_buffer->GetDimension()[1],
      BufferFormat::GrayScale, rgb_buffer->GetOrientation());
  convert_op.SetOutput(output_buffer.get());

  EXPECT_EQ(convert_op.Process(*rgb_buffer), absl::OkStatus());
  EXPECT_EQ(output_buffer->GetBufferFormat(), BufferFormat::GrayScale);

  for (size_t i = 0; i < output_buffer->GetNumElements(); ++i) {
    // we compare use the formula from
    // https://en.wikipeddia.org/wiki/Grayscale
    EXPECT_NEAR((*output_buffer)[0].data[i],
                0.299 * (*rgb_buffer)[0].data[i * 3] +
                    0.587 * (*rgb_buffer)[0].data[i * 3 + 1] +
                    0.114 * (*rgb_buffer)[0].data[i * 3 + 2],
                1);
  }
}

TEST(ImageOperationTest, RotateImageTest) {
  RotateOperation rotate_op(90);
  std::shared_ptr<Buffer> input_buffer = LoadImage("band/test/data/hippo.jpg");
  EXPECT_EQ(rotate_op.Process(*input_buffer), absl::OkStatus());
  auto output_buffer = rotate_op.GetOutput();

  // the dimension should be swapped
  EXPECT_EQ(output_buffer->GetDimension()[0], input_buffer->GetDimension()[1]);
  EXPECT_EQ(output_buffer->GetDimension()[1], input_buffer->GetDimension()[0]);
  // rotate back to original
  RotateOperation rotate_back_op(270);
  EXPECT_EQ(rotate_back_op.Process(*output_buffer), absl::OkStatus());
  auto output_buffer2 = rotate_back_op.GetOutput();

  EXPECT_EQ(output_buffer2->GetDimension()[0], input_buffer->GetDimension()[0]);
  EXPECT_EQ(output_buffer2->GetDimension()[1], input_buffer->GetDimension()[1]);

  for (size_t i = 0; i < output_buffer2->GetNumElements(); ++i) {
    EXPECT_EQ((*output_buffer2)[0].data[i], (*input_buffer)[0].data[i]);
  }
}

TEST(ImageOperationTest, ResizeImageTest) {
  ResizeOperation resize_op(256, 256);
  std::shared_ptr<Buffer> input_buffer = LoadImage("band/test/data/hippo.jpg");
  EXPECT_EQ(resize_op.Process(*input_buffer), absl::OkStatus());
  auto output_buffer = resize_op.GetOutput();

  EXPECT_EQ(output_buffer->GetDimension()[0], 256);
  EXPECT_EQ(output_buffer->GetDimension()[1], 256);

  std::shared_ptr<Buffer> resized_buffer =
      LoadImage("band/test/data/hippo_resize_256.jpg");

  // give some tolerance
  for (size_t i = 0; i < output_buffer->GetNumElements(); ++i) {
    EXPECT_NEAR((*output_buffer)[0].data[i], (*resized_buffer)[0].data[i], 3);
  }
}

}  // namespace test
}  // namespace band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}