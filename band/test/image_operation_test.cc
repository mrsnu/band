// Copyright 2023 Seoul National University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "band/buffer/buffer.h"
#include "band/buffer/image_operator.h"
#include "band/test/image_util.h"

namespace band {
using namespace buffer;

namespace test {
TEST(ImageOperationTest, CropOperationSimpleTest) {
  Crop crop_op(0, 0, 1, 1);
  std::array<unsigned char, 9> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto input_buffer =
      Buffer::CreateFromRaw(input_data.data(), 3, 3, BufferFormat::kGrayScale);
  EXPECT_EQ(crop_op.Process(*input_buffer), absl::OkStatus());
  auto output_buffer = crop_op.GetOutput();
  EXPECT_EQ(output_buffer->GetDimension()[0], 2);
  EXPECT_EQ(output_buffer->GetDimension()[1], 2);
  EXPECT_EQ(output_buffer->GetNumPlanes(), 1);
  EXPECT_EQ(output_buffer->GetBufferFormat(), BufferFormat::kGrayScale);
  EXPECT_EQ(output_buffer->GetOrientation(), BufferOrientation::kTopLeft);
  EXPECT_EQ(output_buffer->GetNumElements(), 4);
  EXPECT_EQ((*output_buffer)[0].data[0], 1);
  EXPECT_EQ((*output_buffer)[0].data[1], 2);
  EXPECT_EQ((*output_buffer)[0].data[2], 4);
  EXPECT_EQ((*output_buffer)[0].data[3], 5);
}

TEST(ImageOperationTest, CropOperationImageTest) {
  Crop crop_op(0, 0, 255, 255);
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

TEST(ImageOperationTest, CropOperationFailureTest) {
  // load (598x305) img
  std::shared_ptr<Buffer> input_buffer = LoadImage("band/test/data/hippo.jpg");
  // x1 > width && y1 > height
  Crop crop_out_of_bound(0, 0, 600, 400);
  EXPECT_EQ(crop_out_of_bound.Process(*input_buffer).code(),
            absl::StatusCode::kInvalidArgument);
  // x0 > x1
  Crop crop_out_of_bound2(255, 0, 0, 255);
  EXPECT_EQ(crop_out_of_bound2.Process(*input_buffer).code(),
            absl::StatusCode::kInvalidArgument);
  // y0 > y1
  Crop crop_out_of_bound3(0, 255, 255, 0);
  EXPECT_EQ(crop_out_of_bound3.Process(*input_buffer).code(),
            absl::StatusCode::kInvalidArgument);

  // negative value
  Crop crop_negative(-1, -1, 256, 256);
  EXPECT_EQ(crop_negative.Process(*input_buffer).code(),
            absl::StatusCode::kInvalidArgument);

  // negative value (x1 and y1)
  Crop crop_negative2(0, 0, -1, -1);
  EXPECT_EQ(crop_negative2.Process(*input_buffer).code(),
            absl::StatusCode::kInvalidArgument);

  Crop crop_negative3(-1, -1, -1, -1);
  EXPECT_EQ(crop_negative3.Process(*input_buffer).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(ImageOperationTest, ConvertImageTest) {
  ColorSpaceConvert convert_op;
  // load 3-channel images
  std::shared_ptr<Buffer> rgb_buffer = LoadImage("band/test/data/hippo.jpg");

  EXPECT_EQ(rgb_buffer->GetBufferFormat(), BufferFormat::kRGB);
  // convert to gray scale
  std::shared_ptr<Buffer> output_buffer(Buffer::CreateEmpty(
      rgb_buffer->GetDimension()[0], rgb_buffer->GetDimension()[1],
      BufferFormat::kGrayScale, DataType::kUInt8,
      rgb_buffer->GetOrientation()));
  convert_op.SetOutput(output_buffer.get());

  EXPECT_EQ(convert_op.Process(*rgb_buffer), absl::OkStatus());
  EXPECT_EQ(output_buffer->GetBufferFormat(), BufferFormat::kGrayScale);

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

TEST(ImageOperationTest, ConvertWithoutImageTest) {
  ColorSpaceConvert convert_op(BufferFormat::kGrayScale);
  // load 3-channel images
  std::shared_ptr<Buffer> rgb_buffer = LoadImage("band/test/data/hippo.jpg");
  EXPECT_EQ(convert_op.Process(*rgb_buffer), absl::OkStatus());
  auto output_buffer = convert_op.GetOutput();
  EXPECT_EQ(output_buffer->GetBufferFormat(), BufferFormat::kGrayScale);

  for (size_t i = 0; i < output_buffer->GetNumElements(); ++i) {
    EXPECT_NEAR((*output_buffer)[0].data[i],
                0.299 * (*rgb_buffer)[0].data[i * 3] +
                    0.587 * (*rgb_buffer)[0].data[i * 3 + 1] +
                    0.114 * (*rgb_buffer)[0].data[i * 3 + 2],
                1);
  }
}

TEST(ImageOperationTest, RotateImageTest) {
  Rotate rotate_op(90);
  std::shared_ptr<Buffer> input_buffer = LoadImage("band/test/data/hippo.jpg");
  EXPECT_EQ(rotate_op.Process(*input_buffer), absl::OkStatus());
  auto output_buffer = rotate_op.GetOutput();

  // the dimension should be swapped
  EXPECT_EQ(output_buffer->GetDimension()[0], input_buffer->GetDimension()[1]);
  EXPECT_EQ(output_buffer->GetDimension()[1], input_buffer->GetDimension()[0]);
  // rotate back to original
  Rotate rotate_back_op(270);
  EXPECT_EQ(rotate_back_op.Process(*output_buffer), absl::OkStatus());
  auto output_buffer2 = rotate_back_op.GetOutput();

  EXPECT_EQ(output_buffer2->GetDimension()[0], input_buffer->GetDimension()[0]);
  EXPECT_EQ(output_buffer2->GetDimension()[1], input_buffer->GetDimension()[1]);

  for (size_t i = 0; i < output_buffer2->GetNumElements(); ++i) {
    EXPECT_EQ((*output_buffer2)[0].data[i], (*input_buffer)[0].data[i]);
  }
}

TEST(ImageOperationTest, ResizeImageTest) {
  Resize resize_op(256, 256);
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