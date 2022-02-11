/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/c/c_api.h"

#include <stdarg.h>
#include <stdint.h>

#include <array>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/testing/util.h"

namespace {

TEST(CAPI, Version) { EXPECT_STRNE("", TfLiteVersion()); }

TEST(CApiSimple, Smoke) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsSetConfigPath(options, "tensorflow/lite/testdata/runtime_config.json");

  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(options);
  ASSERT_NE(interpreter, nullptr);

  int32_t model_id = TfLiteInterpreterRegisterModel(interpreter, model);

  // The options/model can be deleted immediately after interpreter creation.
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);

  ASSERT_EQ(TfLiteInterpreterGetInputTensorCount(interpreter, model_id), 1);
  ASSERT_EQ(TfLiteInterpreterGetOutputTensorCount(interpreter, model_id), 1);

  TfLiteTensor* input_tensor = TfLiteInterpreterAllocateInputTensor(interpreter, model_id, 0);
  ASSERT_NE(input_tensor, nullptr);
  EXPECT_EQ(TfLiteTensorType(input_tensor), kTfLiteFloat32);
  EXPECT_EQ(TfLiteTensorNumDims(input_tensor), 1);
  EXPECT_EQ(TfLiteTensorDim(input_tensor, 0), 2);
  EXPECT_EQ(TfLiteTensorByteSize(input_tensor), sizeof(float) * 2);
  EXPECT_NE(TfLiteTensorData(input_tensor), nullptr);
  EXPECT_STREQ(TfLiteTensorName(input_tensor), "input");

  TfLiteQuantizationParams input_params =
      TfLiteTensorQuantizationParams(input_tensor);
  EXPECT_EQ(input_params.scale, 0.f);
  EXPECT_EQ(input_params.zero_point, 0);

  std::array<float, 2> input = {1.f, 3.f};
  ASSERT_EQ(TfLiteTensorCopyFromBuffer(input_tensor, input.data(),
                                       input.size() * sizeof(float)),
            kTfLiteOk);

  TfLiteTensor* output_tensor = TfLiteInterpreterAllocateOutputTensor(interpreter, model_id, 0);

  TfLiteInterpreterInvokeSync(interpreter, model_id, &input_tensor, &output_tensor);

  ASSERT_NE(output_tensor, nullptr);
  EXPECT_EQ(TfLiteTensorType(output_tensor), kTfLiteFloat32);
  EXPECT_EQ(TfLiteTensorNumDims(output_tensor), 1);
  EXPECT_EQ(TfLiteTensorDim(output_tensor, 0), 2);
  EXPECT_EQ(TfLiteTensorByteSize(output_tensor), sizeof(float) * 2);
  EXPECT_NE(TfLiteTensorData(output_tensor), nullptr);
  EXPECT_STREQ(TfLiteTensorName(output_tensor), "output");

  TfLiteQuantizationParams output_params =
      TfLiteTensorQuantizationParams(output_tensor);
  EXPECT_EQ(output_params.scale, 0.f);
  EXPECT_EQ(output_params.zero_point, 0);

  std::array<float, 2> output;
  ASSERT_EQ(TfLiteTensorCopyToBuffer(output_tensor, output.data(),
                                     output.size() * sizeof(float)),
            kTfLiteOk);
  EXPECT_EQ(output[0], 3.f);
  EXPECT_EQ(output[1], 9.f);

  TfLiteInterpreterDelete(interpreter);
}

TEST(CApiSimple, QuantizationParams) {
  TfLiteModel* model = TfLiteModelCreateFromFile(
      "tensorflow/lite/testdata/add_quantized.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsSetConfigPath(options, "tensorflow/lite/testdata/runtime_config.json");

  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(options);
  ASSERT_NE(interpreter, nullptr);

  int32_t model_id = TfLiteInterpreterRegisterModel(interpreter, model);

  // The options/model can be deleted immediately after interpreter creation.
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);

  TfLiteTensor* input_tensor = TfLiteInterpreterAllocateInputTensor(interpreter, model_id, 0);
  ASSERT_NE(input_tensor, nullptr);
  EXPECT_EQ(TfLiteTensorType(input_tensor), kTfLiteUInt8);
  EXPECT_EQ(TfLiteTensorNumDims(input_tensor), 1);
  EXPECT_EQ(TfLiteTensorDim(input_tensor, 0), 2);

  TfLiteQuantizationParams input_params =
      TfLiteTensorQuantizationParams(input_tensor);
  EXPECT_EQ(input_params.scale, 0.003922f);
  EXPECT_EQ(input_params.zero_point, 0);

  const std::array<uint8_t, 2> input = {1, 3};
  ASSERT_EQ(TfLiteTensorCopyFromBuffer(input_tensor, input.data(),
                                       input.size() * sizeof(uint8_t)),
            kTfLiteOk);

  TfLiteTensor* output_tensor = TfLiteInterpreterAllocateOutputTensor(interpreter, model_id, 0);

  TfLiteInterpreterInvokeSync(interpreter, model_id, &input_tensor, &output_tensor);
  ASSERT_NE(output_tensor, nullptr);

  TfLiteQuantizationParams output_params =
      TfLiteTensorQuantizationParams(output_tensor);
  EXPECT_EQ(output_params.scale, 0.003922f);
  EXPECT_EQ(output_params.zero_point, 0);

  std::array<uint8_t, 2> output;
  ASSERT_EQ(TfLiteTensorCopyToBuffer(output_tensor, output.data(),
                                     output.size() * sizeof(uint8_t)),
            kTfLiteOk);
  EXPECT_EQ(output[0], 3);
  EXPECT_EQ(output[1], 9);

  const float dequantizedOutput0 =
      output_params.scale * (output[0] - output_params.zero_point);
  const float dequantizedOutput1 =
      output_params.scale * (output[1] - output_params.zero_point);
  EXPECT_EQ(dequantizedOutput0, 0.011766f);
  EXPECT_EQ(dequantizedOutput1, 0.035298f);

  TfLiteInterpreterDelete(interpreter);
}

TEST(CApiSimple, ValidModel) {
  std::ifstream model_file("tensorflow/lite/testdata/add.bin");

  model_file.seekg(0, std::ios_base::end);
  std::vector<char> model_buffer(model_file.tellg());

  model_file.seekg(0, std::ios_base::beg);
  model_file.read(model_buffer.data(), model_buffer.size());

  TfLiteModel* model =
      TfLiteModelCreate(model_buffer.data(), model_buffer.size());
  ASSERT_NE(model, nullptr);
  TfLiteModelDelete(model);
}

TEST(CApiSimple, ValidModelFromFile) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);
  TfLiteModelDelete(model);
}

TEST(CApiSimple, InvalidModel) {
  std::vector<char> invalid_model(20, 'c');
  TfLiteModel* model =
      TfLiteModelCreate(invalid_model.data(), invalid_model.size());
  ASSERT_EQ(model, nullptr);
}

TEST(CApiSimple, InvalidModelFromFile) {
  TfLiteModel* model = TfLiteModelCreateFromFile("invalid/path/foo.tflite");
  ASSERT_EQ(model, nullptr);
}

}  // namespace

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
