#include <gtest/gtest.h>
#include <stdarg.h>
#include <stdint.h>

#include <array>
#include <fstream>
#include <vector>

#include "band/backend/tfl/interpreter.h"
#include "band/backend/tfl/model.h"
#include "band/backend/tfl/tensor.h"
#include "band/engine.h"
#include "band/interface/backend_factory.h"
#include "band/model.h"
#include "band/tensor.h"

namespace Band {
using namespace Interface;
TEST(TFLiteBackend, BackendInvoke) {
  TfLite::TfLiteModel bin_model(0);
  bin_model.FromPath("band/testdata/add.bin");

  TfLite::TfLiteInterpreter interpreter;
  EXPECT_EQ(interpreter.FromModel(&bin_model, 0, kBandCPU), kBandOk);

  SubgraphKey key(bin_model.GetId(), 0);
  EXPECT_EQ(interpreter.InvokeSubgraph(key), kBandOk);
}

TEST(TFLiteBackend, ModelSpec) {
  TfLite::TfLiteModel bin_model(0);
  bin_model.FromPath("band/testdata/add.bin");

  TfLite::TfLiteInterpreter interpreter;
  ModelSpec model_spec;
  model_spec = interpreter.InvestigateModelSpec(&bin_model);

  EXPECT_EQ(model_spec.num_ops, 2);
  EXPECT_EQ(model_spec.input_tensors.size(), 1);
  EXPECT_EQ(model_spec.output_tensors.size(), 1);
}

TEST(TFLiteBackend, Registration) {
  auto backends = BackendFactory::GetAvailableBackends();

  EXPECT_EQ(backends.size(), 1);
}

TEST(TFLiteBackend, InterfaceInvoke) {
  auto backends = BackendFactory::GetAvailableBackends();
  IModel *bin_model = BackendFactory::CreateModel(kBandTfLite, 0);
  bin_model->FromPath("band/testdata/add.bin");

  IInterpreter *interpreter = BackendFactory::CreateInterpreter(kBandTfLite);
  EXPECT_EQ(interpreter->FromModel(bin_model, 0, kBandCPU), kBandOk);

  SubgraphKey key = interpreter->GetModelSubgraphKey(bin_model->GetId());

  EXPECT_EQ(interpreter->GetInputs(key).size(), 1);
  EXPECT_EQ(interpreter->GetOutputs(key).size(), 1);

  std::array<float, 2> input = {1.f, 3.f};
  memcpy(interpreter->GetTensorView(key, interpreter->GetInputs(key)[0])
             ->GetData(),
         input.data(), input.size() * sizeof(float));

  EXPECT_EQ(interpreter->InvokeSubgraph(key), kBandOk);

  auto output_tensor =
      interpreter->GetTensorView(key, interpreter->GetOutputs(key)[0]);
  EXPECT_EQ(reinterpret_cast<float *>(output_tensor->GetData())[0], 3.f);
  EXPECT_EQ(reinterpret_cast<float *>(output_tensor->GetData())[1], 9.f);

  delete bin_model;
  delete interpreter;
}

TEST(TFLiteBackend, SimpleEngineInvokeSync) {
  RuntimeConfig config;
  EXPECT_EQ(ParseRuntimeConfigFromJson("band/testdata/config.json", config),
            kBandOk);

  auto engine = Engine::Create(config);
  EXPECT_TRUE(engine);

  Model model;
  EXPECT_EQ(model.FromPath(kBandTfLite, "band/testdata/add.bin"), kBandOk);
  EXPECT_EQ(engine->RegisterModel(&model), kBandOk);

  Tensor *input_tensor = engine->CreateTensor(
      model.GetId(), engine->GetInputTensorIndices(model.GetId())[0]);
  Tensor *output_tensor = engine->CreateTensor(
      model.GetId(), engine->GetOutputTensorIndices(model.GetId())[0]);

  EXPECT_TRUE(input_tensor && output_tensor);

  std::array<float, 2> input = {1.f, 3.f};
  memcpy(input_tensor->GetData(), input.data(), input.size() * sizeof(float));

  EXPECT_EQ(
      engine->InvokeSyncModel(model.GetId(), {input_tensor}, {output_tensor}),
      kBandOk);
  EXPECT_EQ(reinterpret_cast<float *>(output_tensor->GetData())[0], 3.f);
  EXPECT_EQ(reinterpret_cast<float *>(output_tensor->GetData())[1], 9.f);

  delete input_tensor;
  delete output_tensor;
}

TEST(TFLiteBackend, SimpleEngineInvokeAsync) {
  RuntimeConfig config;
  EXPECT_EQ(ParseRuntimeConfigFromJson("band/testdata/config.json", config),
            kBandOk);

  auto engine = Engine::Create(config);
  EXPECT_TRUE(engine);

  Model model;
  EXPECT_EQ(model.FromPath(kBandTfLite, "band/testdata/add.bin"), kBandOk);
  EXPECT_EQ(engine->RegisterModel(&model), kBandOk);

  Tensor *input_tensor = engine->CreateTensor(
      model.GetId(), engine->GetInputTensorIndices(model.GetId())[0]);
  Tensor *output_tensor = engine->CreateTensor(
      model.GetId(), engine->GetOutputTensorIndices(model.GetId())[0]);

  EXPECT_TRUE(input_tensor && output_tensor);

  std::array<float, 2> input = {1.f, 3.f};
  memcpy(input_tensor->GetData(), input.data(), input.size() * sizeof(float));

  JobId job_id = engine->InvokeAsyncModel(model.GetId(), {input_tensor});
  EXPECT_EQ(engine->Wait(job_id, {output_tensor}), kBandOk);
  EXPECT_EQ(reinterpret_cast<float *>(output_tensor->GetData())[0], 3.f);
  EXPECT_EQ(reinterpret_cast<float *>(output_tensor->GetData())[1], 9.f);

  delete input_tensor;
  delete output_tensor;
} // namespace
} // namespace Band

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
