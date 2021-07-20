/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/testing/tflite_driver.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/util.h"

#define SAMPLE_INPUT_0 "0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4"
#define SAMPLE_INPUT_1 "0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.04"
#define SAMPLE_INPUT_2 "0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004,0.001,0.002,0.003,0.004"
#define SAMPLE_OUTPUT_0 "0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404,0.101,0.202,0.303,0.404"
#define SAMPLE_OUTPUT_1 "0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044,0.011,0.022,0.033,0.044"

namespace tflite {
namespace testing {
namespace {

using ::testing::ElementsAre;

TEST(TfliteDriverTest, SimpleTest) {
  std::unique_ptr<TestRunner> runner(new TfLiteDriver());
  runner->ResetInterpreter();

  runner->SetModelBaseDir("tensorflow/lite");
  int model_id = runner->LoadModel("testdata/multi_add.bin");
  ASSERT_TRUE(model_id >= 0);
  ASSERT_TRUE(runner->IsValid());

  ASSERT_THAT(runner->GetInputs(model_id), ElementsAre(0, 1, 2, 3));
  ASSERT_THAT(runner->GetOutputs(model_id), ElementsAre(5, 6));

  ASSERT_TRUE(runner->IsValid());

  runner->AllocateTensors(model_id);
  
  runner->SetInput(model_id, 0, SAMPLE_INPUT_0);
  runner->SetInput(model_id, 1, SAMPLE_INPUT_2);
  runner->SetInput(model_id, 2, SAMPLE_INPUT_2);
  runner->SetInput(model_id, 3, SAMPLE_INPUT_1);

  runner->ResetTensor(model_id, 2);

  runner->SetExpectation(model_id, 5, SAMPLE_OUTPUT_0);
  runner->SetExpectation(model_id, 6, SAMPLE_OUTPUT_1);

  runner->Invoke(model_id);
  ASSERT_TRUE(runner->IsValid());

  ASSERT_TRUE(runner->CheckResults(model_id));
  EXPECT_EQ(runner->ReadOutput(model_id, 5), SAMPLE_OUTPUT_0);
  EXPECT_EQ(runner->ReadOutput(model_id, 6), SAMPLE_OUTPUT_1);
}

TEST(TfliteDriverTest, PlannerTest) {
  std::unique_ptr<TestRunner> runner(new TfLiteDriver());
  runner->ResetInterpreter();

  runner->SetModelBaseDir("tensorflow/lite");
  int model_id = runner->LoadModel("testdata/multi_add.bin");
  ASSERT_TRUE(model_id == 0);
  ASSERT_TRUE(runner->IsValid());

  ASSERT_THAT(runner->GetInputs(model_id), ElementsAre(0, 1, 2, 3));
  ASSERT_THAT(runner->GetOutputs(model_id), ElementsAre(5, 6));

  ASSERT_TRUE(runner->IsValid());

  runner->AllocateTensors(model_id);

  runner->SetInput(model_id, 0, SAMPLE_INPUT_0);
  runner->SetInput(model_id, 1, SAMPLE_INPUT_2);
  runner->SetInput(model_id, 2, SAMPLE_INPUT_2);
  runner->SetInput(model_id, 3, SAMPLE_INPUT_1);

  runner->ResetTensor(model_id, 2);

  runner->SetExpectation(model_id, 5, SAMPLE_OUTPUT_0);
  runner->SetExpectation(model_id, 6, SAMPLE_OUTPUT_1);

  runner->InvokeThroughPlanner(model_id);
  ASSERT_TRUE(runner->IsValid());

  ASSERT_TRUE(runner->CheckResults(model_id));
  EXPECT_EQ(runner->ReadOutput(model_id, 5), SAMPLE_OUTPUT_0);
  EXPECT_EQ(runner->ReadOutput(model_id, 6), SAMPLE_OUTPUT_1);
}

TEST(TfliteDriverTest, AddQuantizedInt8Test) {
  std::unique_ptr<TestRunner> runner(new TfLiteDriver());
  runner->ResetInterpreter();

  runner->SetModelBaseDir("tensorflow/lite");
  int model_id = runner->LoadModel("testdata/add_quantized_int8.bin");
  ASSERT_TRUE(model_id >= 0);
  ASSERT_TRUE(runner->IsValid());

  ASSERT_THAT(runner->GetInputs(model_id), ElementsAre(1));
  ASSERT_THAT(runner->GetOutputs(model_id), ElementsAre(2));

  runner->ReshapeTensor(model_id, 1, "1,2,2,1");
  ASSERT_TRUE(runner->IsValid());

  runner->AllocateTensors(model_id);

  runner->SetInput(model_id, 1, "1,1,1,1");

  runner->SetExpectation(model_id, 2, "0.0117,0.0117,0.0117,0.0117");

  runner->Invoke(model_id);
  ASSERT_TRUE(runner->IsValid());

  ASSERT_TRUE(runner->CheckResults(model_id));
  EXPECT_EQ(runner->ReadOutput(model_id, 2), "3,3,3,3");
}

TEST(TfliteDriverTest, RingBufferTest) {
  // Checks if copying to the input ring buffer and output ring buffer works correctly
  std::unique_ptr<TestRunner> runner(new TfLiteDriver());
  runner->ResetInterpreter();

  runner->SetModelBaseDir("tensorflow/lite");
  int model_id = runner->LoadModel("testdata/multi_add.bin");
  ASSERT_TRUE(model_id == 0);
  ASSERT_TRUE(runner->IsValid());

  ASSERT_THAT(runner->GetInputs(model_id), ElementsAre(0, 1, 2, 3));
  ASSERT_THAT(runner->GetOutputs(model_id), ElementsAre(5, 6));

  ASSERT_TRUE(runner->IsValid());

  std::vector<tflite::Job> jobs;
  jobs.push_back(tflite::Job({ model_id }));

  std::vector<tflite::Tensors> inputs;
  std::vector<tflite::Tensors> outputs;
  tflite::Tensors input_tensors;
  tflite::Tensors output_tensors;

  // input tensor allocate
  for (int i : {0, 1, 2, 3}) {
    input_tensors.push_back(runner->AllocateInputTensor(model_id, i));
  }
  // output tensor allocate
  for (int i : {0, 1}) {
    output_tensors.push_back(runner->AllocateOutputTensor(model_id, i));
  }

  runner->SetDataToTensor(input_tensors[0], SAMPLE_INPUT_0);
  runner->SetDataToTensor(input_tensors[1], SAMPLE_INPUT_2);
  runner->SetDataToTensor(input_tensors[2], SAMPLE_INPUT_2);
  runner->SetDataToTensor(input_tensors[3], SAMPLE_INPUT_1);

  runner->ResetTensor(input_tensors[2]);

  ASSERT_TRUE(runner->IsValid());

  inputs.push_back(input_tensors);
  outputs.push_back(output_tensors);

  runner->InvokeWithInput(jobs, inputs, outputs);
  ASSERT_TRUE(runner->IsValid());

  ASSERT_TRUE(runner->CheckResults(model_id));
  EXPECT_EQ(runner->ReadOutput(output_tensors[0]), SAMPLE_OUTPUT_0);
  EXPECT_EQ(runner->ReadOutput(output_tensors[1]), SAMPLE_OUTPUT_1);
}

}  // namespace
}  // namespace testing
}  // namespace tflite
