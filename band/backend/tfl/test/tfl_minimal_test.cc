#include <gtest/gtest.h>
#include <stdarg.h>
#include <stdint.h>

#include <array>
#include <fstream>
#include <vector>

#include "band/backend/tfl/interpreter.h"
#include "band/backend/tfl/model.h"
#include "band/backend/tfl/tensor.h"
#include "band/backend_factory.h"
#include "band/config_builder.h"
#include "band/engine.h"
#include "band/model.h"
#include "band/tensor.h"

namespace Band {
using namespace Interface;
TEST(TFLiteBackend, BackendInvoke) {
  TfLite::TfLiteModel bin_model(0);
  bin_model.FromPath("band/testdata/add.bin");

  TfLite::TfLiteInterpreter interpreter;
  auto key = interpreter.FromModel(&bin_model, 0, kBandCPU);
  EXPECT_TRUE(key.ok());
  EXPECT_TRUE(interpreter.InvokeSubgraph(key.value()).ok());
}

TEST(TFLiteBackend, ModelSpec) {
  TfLite::TfLiteModel bin_model(0);
  bin_model.FromPath("band/testdata/add.bin");

  TfLite::TfLiteInterpreter interpreter;
  ModelSpec model_spec;
  model_spec = interpreter.InvestigateModelSpec(&bin_model);

  EXPECT_EQ(model_spec.num_ops, 1);
  EXPECT_EQ(model_spec.input_tensors.size(), 1);
  EXPECT_EQ(model_spec.output_tensors.size(), 1);
}

TEST(TFLiteBackend, Registration) {
  auto backends = BackendFactory::GetAvailableBackends();

  EXPECT_EQ(backends.size(), 1);
}

TEST(TFLiteBackend, InterfaceInvoke) {
  auto backends = BackendFactory::GetAvailableBackends();
  auto bin_model_status = BackendFactory::CreateModel(kBandTfLite, 0);
  EXPECT_TRUE(bin_model_status.ok());
  auto bin_model = bin_model_status.value();
  bin_model->FromPath("band/testdata/add.bin");

  auto interpreter_status = BackendFactory::CreateInterpreter(kBandTfLite);
  EXPECT_TRUE(interpreter_status.ok());
  auto interpreter = interpreter_status.value();

  auto key = interpreter->FromModel(bin_model, 0, kBandCPU);
  EXPECT_TRUE(key.ok());
  EXPECT_EQ(interpreter->GetInputs(key.value()).size(), 1);
  EXPECT_EQ(interpreter->GetOutputs(key.value()).size(), 1);

  std::array<float, 2> input = {1.f, 3.f};
  memcpy(
      interpreter
          ->GetTensorView(key.value(), interpreter->GetInputs(key.value())[0])
          ->GetData(),
      input.data(), input.size() * sizeof(float));

  EXPECT_TRUE(interpreter->InvokeSubgraph(key.value()).ok());

  auto output_tensor = interpreter->GetTensorView(
      key.value(), interpreter->GetOutputs(key.value())[0]);
  EXPECT_EQ(reinterpret_cast<float*>(output_tensor->GetData())[0], 3.f);
  EXPECT_EQ(reinterpret_cast<float*>(output_tensor->GetData())[1], 9.f);

  delete bin_model;
  delete interpreter;
}

TEST(TFLiteBackend, SimpleEngineInvokeSync) {
  RuntimeConfigBuilder b;
  RuntimeConfig config = b.AddPlannerLogPath("band/testdata/log.csv")
                             .AddSchedulers({kBandRoundRobin})
                             .AddMinimumSubgraphSize(7)
                             .AddSubgraphPreparationType(kBandMergeUnitSubgraph)
                             .AddCPUMask(kBandAll)
                             .AddPlannerCPUMask(kBandPrimary)
                             .AddWorkers({kBandCPU, kBandCPU})
                             .AddWorkerNumThreads({3, 4})
                             .AddWorkerCPUMasks({kBandBig, kBandLittle})
                             .AddSmoothingFactor(0.1)
                             .AddProfileDataPath("band/testdata/profile.json")
                             .AddOnline(true)
                             .AddNumWarmups(1)
                             .AddNumRuns(1)
                             .AddAllowWorkSteal(true)
                             .AddAvailabilityCheckIntervalMs(30000)
                             .AddScheduleWindowSize(10)
                             .Build();

  auto engine_status = Engine::Create(config);
  EXPECT_TRUE(engine_status.ok());
  auto engine = std::move(engine_status.value());

  Model model;
  EXPECT_TRUE(model.FromPath(kBandTfLite, "band/testdata/add.bin").ok());
  EXPECT_TRUE(engine->RegisterModel(&model).ok());

  auto input_idx = engine->GetInputTensorIndices(model.GetId());
  auto output_idx = engine->GetOutputTensorIndices(model.GetId());
  EXPECT_TRUE(input_idx.ok() && output_idx.ok());
  auto input_tensor = engine->CreateTensor(model.GetId(), input_idx.value()[0]);
  auto output_tensor =
      engine->CreateTensor(model.GetId(), output_idx.value()[0]);
  EXPECT_TRUE(input_tensor.ok() && output_tensor.ok());

  std::array<float, 2> input = {1.f, 3.f};
  memcpy(input_tensor.value()->GetData(), input.data(),
         input.size() * sizeof(float));

  EXPECT_TRUE(engine
                  ->InvokeSyncModel(model.GetId(), {input_tensor.value()},
                                    {output_tensor.value()})
                  .ok());
  EXPECT_EQ(reinterpret_cast<float*>(output_tensor.value()->GetData())[0], 3.f);
  EXPECT_EQ(reinterpret_cast<float*>(output_tensor.value()->GetData())[1], 9.f);

  delete input_tensor.value();
  delete output_tensor.value();
}

TEST(TFLiteBackend, SimpleEngineInvokeAsync) {
  RuntimeConfigBuilder b;
  RuntimeConfig config = b.AddPlannerLogPath("band/testdata/log.csv")
                             .AddSchedulers({kBandRoundRobin})
                             .AddMinimumSubgraphSize(7)
                             .AddSubgraphPreparationType(kBandMergeUnitSubgraph)
                             .AddCPUMask(kBandAll)
                             .AddPlannerCPUMask(kBandPrimary)
                             .AddWorkers({kBandCPU, kBandCPU})
                             .AddWorkerNumThreads({3, 4})
                             .AddWorkerCPUMasks({kBandBig, kBandLittle})
                             .AddSmoothingFactor(0.1)
                             .AddProfileDataPath("band/testdata/profile.json")
                             .AddOnline(true)
                             .AddNumWarmups(1)
                             .AddNumRuns(1)
                             .AddAllowWorkSteal(true)
                             .AddAvailabilityCheckIntervalMs(30000)
                             .AddScheduleWindowSize(10)
                             .Build();

  auto engine_status = Engine::Create(config);
  EXPECT_TRUE(engine_status.ok());
  auto engine = std::move(engine_status.value());

  Model model;
  EXPECT_TRUE(model.FromPath(kBandTfLite, "band/testdata/add.bin").ok());
  EXPECT_TRUE(engine->RegisterModel(&model).ok());

  auto input_idx = engine->GetInputTensorIndices(model.GetId());
  auto output_idx = engine->GetOutputTensorIndices(model.GetId());
  EXPECT_TRUE(input_idx.ok() && output_idx.ok());
  auto input_tensor = engine->CreateTensor(model.GetId(), input_idx.value()[0]);
  auto output_tensor =
      engine->CreateTensor(model.GetId(), output_idx.value()[0]);
  EXPECT_TRUE(input_tensor.ok() && output_tensor.ok());

  std::array<float, 2> input = {1.f, 3.f};
  memcpy(input_tensor.value()->GetData(), input.data(), input.size() * sizeof(float));

  JobId job_id = engine->InvokeAsyncModel(model.GetId(), {input_tensor.value()});
  EXPECT_TRUE(engine->Wait(job_id, {output_tensor.value()}).ok());
  EXPECT_EQ(reinterpret_cast<float*>(output_tensor.value()->GetData())[0], 3.f);
  EXPECT_EQ(reinterpret_cast<float*>(output_tensor.value()->GetData())[1], 9.f);

  delete input_tensor.value();
  delete output_tensor.value();
}  // namespace
}  // namespace Band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
