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

#include "tensorflow/lite/c/c_api_experimental.h"

#include <gtest/gtest.h>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/testing/util.h"

namespace {

TfLiteRegistration* GetDummyRegistration() {
  static TfLiteRegistration registration = {
      /*init=*/nullptr,
      /*free=*/nullptr,
      /*prepare=*/nullptr,
      /*invoke=*/[](TfLiteContext*, TfLiteNode*) { return kTfLiteOk; }};
  return &registration;
}

TEST(CApiExperimentalTest, Smoke) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsSetConfigPath(options, "tensorflow/lite/testdata/runtime_config.json")

  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(options);
  TfLiteModelAddBuiltinOp(interpreter, kTfLiteBuiltinAdd,
                                       GetDummyRegistration(), 1, 1);
  ASSERT_NE(interpreter, nullptr);
  int32_t model_id = TfLiteInterpreterRegisterModel(interpreter, model);

  std::vector<TfLiteTensor*> input_tensors;
  for (int i = 0; i < TfLiteInterpreterGetInputTensorCount(interpreter, model_id); i++) {
    input_tensors.push_back(TfLiteInterpreterAllocateInputTensor(interpreter, model_id, i));
  }

  TfLiteTensor* output_tensor = TfLiteInterpreterAllocateOutputTensor(interpreter, model_id, 0);

  TfLiteInterpreterInvokeSync(interpreter, model_id, input_tensors.data(), output_tensor);

  TfLiteInterpreterDelete(interpreter);
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);

  for (int i = 0; i < TfLiteInterpreterGetInputTensorCount(interpreter, model_id); i++) {
    TfLiteInterpreterDeleteTensor(input_tensors[i]);
  }
  TfLiteInterpreterDeleteTensor(output_tensor);
}

}  // namespace

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
