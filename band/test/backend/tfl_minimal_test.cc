#include <gtest/gtest.h>
#include <stdarg.h>
#include <stdint.h>

#include <array>
#include <fstream>
#include <vector>

#include "band/backend/tfl/model.h"
#include "band/backend/tfl/model_executor.h"
#include "band/backend/tfl/tensor.h"
#include "band/backend_factory.h"
#include "band/config_builder.h"
#include "band/engine.h"
#include "band/interface/model_executor.h"
#include "band/interface/tensor.h"
#include "band/model.h"
#include "band/tensor.h"

namespace Band {
using namespace Interface;
TEST(TFLiteBackend, BackendInvoke) {
  TfLite::TfLiteModel bin_model(0);
  bin_model.FromPath("band/test/data/add.tflite");

  TfLite::TfLiteModelExecutor model_executor(0, 0, kBandCPU);
  EXPECT_EQ(model_executor.PrepareSubgraph(&bin_model), kBandOk);
  EXPECT_EQ(
      model_executor.ExecuteSubgraph(model_executor.GetLargestSubgraphKey()),
      kBandOk);
}

TEST(TFLiteBackend, ModelSpec) {
  TfLite::TfLiteModel bin_model(0);
  bin_model.FromPath("band/test/data/add.tflite");

  TfLite::TfLiteModelExecutor model_executor(0, 0, kBandCPU);
  ModelSpec model_spec = model_executor.InvestigateModelSpec(&bin_model);

#ifdef TFLITE_BUILD_WITH_XNNPACK_DELEGATE
  EXPECT_EQ(model_spec.num_ops, 1);
#else
  EXPECT_EQ(model_spec.num_ops, 2);
#endif
  EXPECT_EQ(model_spec.input_tensors.size(), 1);
  EXPECT_EQ(model_spec.output_tensors.size(), 1);
}

TEST(TFLiteBackend, Registration) {
  auto backends = BackendFactory::GetAvailableBackends();
  int expected_num_backends = 0;
#ifdef BAND_TFLITE
  expected_num_backends++;
#endif
  EXPECT_EQ(backends.size(), expected_num_backends);
}

TEST(TFLiteBackend, InterfaceInvoke) {
  auto backends = BackendFactory::GetAvailableBackends();
  IModel* bin_model = BackendFactory::CreateModel(BackendType::TfLite, 0);
  bin_model->FromPath("band/test/data/add.tflite");

  IModelExecutor* model_executor =
      BackendFactory::CreateModelExecutor(BackendType::TfLite, 0, 0, kBandCPU);
  EXPECT_EQ(model_executor->PrepareSubgraph(bin_model), kBandOk);

  SubgraphKey key = model_executor->GetLargestSubgraphKey();

  EXPECT_EQ(model_executor->GetInputs(key).size(), 1);
  EXPECT_EQ(model_executor->GetOutputs(key).size(), 1);

  std::array<float, 2> input = {1.f, 3.f};
  memcpy(model_executor->GetTensorView(key, model_executor->GetInputs(key)[0])
             ->GetData(),
         input.data(), input.size() * sizeof(float));

  EXPECT_EQ(model_executor->ExecuteSubgraph(key), kBandOk);

  auto output_tensor =
      model_executor->GetTensorView(key, model_executor->GetOutputs(key)[0]);
  EXPECT_EQ(reinterpret_cast<float*>(output_tensor->GetData())[0], 3.f);
  EXPECT_EQ(reinterpret_cast<float*>(output_tensor->GetData())[1], 9.f);

  delete bin_model;
  delete model_executor;
}

TEST(TFLiteBackend, SimpleEngineInvokeSync) {
  RuntimeConfigBuilder b;
  RuntimeConfig config = b.AddPlannerLogPath("band/test/data/log.csv")
                             .AddSchedulers({kBandRoundRobin})
                             .AddMinimumSubgraphSize(7)
                             .AddSubgraphPreparationType(kBandMergeUnitSubgraph)
                             .AddCPUMask(kBandAll)
                             .AddPlannerCPUMask(kBandPrimary)
                             .AddWorkers({kBandCPU, kBandCPU})
                             .AddWorkerNumThreads({3, 4})
                             .AddWorkerCPUMasks({kBandBig, kBandLittle})
                             .AddSmoothingFactor(0.1)
                             .AddProfileDataPath("band/test/data/profile.json")
                             .AddOnline(true)
                             .AddNumWarmups(1)
                             .AddNumRuns(1)
                             .AddAllowWorkSteal(true)
                             .AddAvailabilityCheckIntervalMs(30000)
                             .AddScheduleWindowSize(10)
                             .Build();

  auto engine = Engine::Create(config);
  EXPECT_TRUE(engine);

  Model model;
  EXPECT_EQ(model.FromPath(BackendType::TfLite, "band/test/data/add.tflite"), kBandOk);
  EXPECT_EQ(engine->RegisterModel(&model), kBandOk);

  Tensor* input_tensor = engine->CreateTensor(
      model.GetId(), engine->GetInputTensorIndices(model.GetId())[0]);
  Tensor* output_tensor = engine->CreateTensor(
      model.GetId(), engine->GetOutputTensorIndices(model.GetId())[0]);

  EXPECT_TRUE(input_tensor && output_tensor);

  int execution_count = 0;
  engine->SetOnEndRequest(
      [&execution_count](int, BandStatus) { execution_count++; });

  std::array<float, 2> input = {1.f, 3.f};
  memcpy(input_tensor->GetData(), input.data(), input.size() * sizeof(float));

  EXPECT_EQ(engine->RequestSync(model.GetId(), BandGetDefaultRequestOption(),
                                {input_tensor}, {output_tensor}),
            kBandOk);
  EXPECT_EQ(reinterpret_cast<float*>(output_tensor->GetData())[0], 3.f);
  EXPECT_EQ(reinterpret_cast<float*>(output_tensor->GetData())[1], 9.f);
  EXPECT_EQ(execution_count, 1);

  delete input_tensor;
  delete output_tensor;
}

TEST(TFLiteBackend, SimpleEngineProfile) {
  RuntimeConfigBuilder b;
  RuntimeConfig config = b.AddPlannerLogPath("band/test/data/log.csv")
                             .AddSchedulers({kBandFixedWorkerGlobalQueue})
                             .AddMinimumSubgraphSize(7)
                             .AddSubgraphPreparationType(kBandMergeUnitSubgraph)
                             .AddCPUMask(kBandAll)
                             .AddPlannerCPUMask(kBandPrimary)
                             .AddWorkers({kBandCPU, kBandCPU})
                             .AddWorkerNumThreads({3, 4})
                             .AddWorkerCPUMasks({kBandBig, kBandLittle})
                             .AddSmoothingFactor(0.1)
                             .AddProfileDataPath("band/test/data/profile.json")
                             .AddOnline(true)
                             .AddNumWarmups(1)
                             .AddNumRuns(1)
                             .AddAllowWorkSteal(true)
                             .AddAvailabilityCheckIntervalMs(30000)
                             .AddScheduleWindowSize(10)
                             .Build();

  auto engine = Engine::Create(config);
  EXPECT_TRUE(engine);

  Model model;
  EXPECT_EQ(model.FromPath(BackendType::TfLite, "band/test/data/add.tflite"), kBandOk);
  EXPECT_EQ(engine->RegisterModel(&model), kBandOk);

  EXPECT_GT(
      engine->GetProfiled(engine->GetLargestSubgraphKey(model.GetId(), 0)), 0);
  EXPECT_GT(
      engine->GetExpected(engine->GetLargestSubgraphKey(model.GetId(), 0)), 0);
}

TEST(TFLiteBackend, SimpleEngineInvokeAsync) {
  RuntimeConfigBuilder b;
  RuntimeConfig config = b.AddPlannerLogPath("band/test/data/log.csv")
                             .AddSchedulers({kBandShortestExpectedLatency})
                             .AddMinimumSubgraphSize(7)
                             .AddSubgraphPreparationType(kBandMergeUnitSubgraph)
                             .AddCPUMask(kBandAll)
                             .AddPlannerCPUMask(kBandPrimary)
                             .AddWorkers({kBandCPU, kBandCPU})
                             .AddWorkerNumThreads({3, 4})
                             .AddWorkerCPUMasks({kBandBig, kBandLittle})
                             .AddSmoothingFactor(0.1)
                             .AddProfileDataPath("band/test/data/profile.json")
                             .AddOnline(true)
                             .AddNumWarmups(1)
                             .AddNumRuns(1)
                             .AddAllowWorkSteal(true)
                             .AddAvailabilityCheckIntervalMs(30000)
                             .AddScheduleWindowSize(10)
                             .Build();

  auto engine = Engine::Create(config);
  EXPECT_TRUE(engine);

  Model model;
  EXPECT_EQ(model.FromPath(BackendType::TfLite, "band/test/data/add.tflite"), kBandOk);
  EXPECT_EQ(engine->RegisterModel(&model), kBandOk);

  Tensor* input_tensor = engine->CreateTensor(
      model.GetId(), engine->GetInputTensorIndices(model.GetId())[0]);
  Tensor* output_tensor = engine->CreateTensor(
      model.GetId(), engine->GetOutputTensorIndices(model.GetId())[0]);

  EXPECT_TRUE(input_tensor && output_tensor);

  std::array<float, 2> input = {1.f, 3.f};
  memcpy(input_tensor->GetData(), input.data(), input.size() * sizeof(float));

  int execution_count = 0;
  engine->SetOnEndRequest(
      [&execution_count](int, BandStatus) { execution_count++; });

  JobId job_id = engine->RequestAsync(
      model.GetId(), BandGetDefaultRequestOption(), {input_tensor});
  EXPECT_EQ(engine->Wait(job_id, {output_tensor}), kBandOk);
  EXPECT_EQ(reinterpret_cast<float*>(output_tensor->GetData())[0], 3.f);
  EXPECT_EQ(reinterpret_cast<float*>(output_tensor->GetData())[1], 9.f);
  EXPECT_EQ(execution_count, 1);

  delete input_tensor;
  delete output_tensor;
}  // namespace

TEST(TFLiteBackend, SimpleEngineInvokeSyncOnWorker) {
  RuntimeConfigBuilder b;
  RuntimeConfig config =
      b.AddPlannerLogPath("band/test/data/log.csv")
          .AddSchedulers({kBandFixedWorker})
          .AddMinimumSubgraphSize(7)
          .AddSubgraphPreparationType(kBandMergeUnitSubgraph)
          .AddCPUMask(kBandAll)
          .AddPlannerCPUMask(kBandPrimary)
          .AddWorkers({kBandCPU, kBandCPU, kBandDSP, kBandNPU, kBandGPU})
          .AddWorkerNumThreads({3, 4, 1, 1, 1})
          .AddWorkerCPUMasks(
              {kBandBig, kBandLittle, kBandAll, kBandAll, kBandAll})
          .AddSmoothingFactor(0.1)
          .AddProfileDataPath("band/test/data/profile.json")
          .AddOnline(true)
          .AddNumWarmups(1)
          .AddNumRuns(1)
          .AddAllowWorkSteal(true)
          .AddAvailabilityCheckIntervalMs(30000)
          .AddScheduleWindowSize(10)
          .Build();

  auto engine = Engine::Create(config);
  EXPECT_TRUE(engine);

  Model model;
  EXPECT_EQ(model.FromPath(BackendType::TfLite, "band/test/data/add.tflite"), kBandOk);
  EXPECT_EQ(engine->RegisterModel(&model), kBandOk);

  Tensor* input_tensor = engine->CreateTensor(
      model.GetId(), engine->GetInputTensorIndices(model.GetId())[0]);
  Tensor* output_tensor = engine->CreateTensor(
      model.GetId(), engine->GetOutputTensorIndices(model.GetId())[0]);

  EXPECT_TRUE(input_tensor && output_tensor);

  std::array<float, 2> input = {1.f, 3.f};
  memcpy(input_tensor->GetData(), input.data(), input.size() * sizeof(float));

  std::cout << "Num workers " << engine->GetNumWorkers() << std::endl;
  for (int worker_id = 0; worker_id < engine->GetNumWorkers(); worker_id++) {
    std::cout << "Run on worker (device: "
              << BandDeviceGetName(engine->GetWorkerDevice(worker_id)) << ")"
              << std::endl;
    EXPECT_EQ(engine->RequestSync(model.GetId(), {worker_id, true, -1, -1},
                                  {input_tensor}, {output_tensor}),
              kBandOk);
    EXPECT_EQ(reinterpret_cast<float*>(output_tensor->GetData())[0], 3.f);
    EXPECT_EQ(reinterpret_cast<float*>(output_tensor->GetData())[1], 9.f);

    memset(output_tensor->GetData(), 0, sizeof(float) * 2);
  }

  delete input_tensor;
  delete output_tensor;
}

TEST(TFLiteBackend, SimpleEngineInvokeCallback) {
  RuntimeConfigBuilder b;
  RuntimeConfig config =
      b.AddPlannerLogPath("band/test/data/log.csv")
          .AddSchedulers({kBandFixedWorker})
          .AddMinimumSubgraphSize(7)
          .AddSubgraphPreparationType(kBandMergeUnitSubgraph)
          .AddCPUMask(kBandAll)
          .AddPlannerCPUMask(kBandPrimary)
          .AddWorkers({kBandCPU, kBandCPU, kBandDSP, kBandNPU, kBandGPU})
          .AddWorkerNumThreads({3, 4, 1, 1, 1})
          .AddWorkerCPUMasks(
              {kBandBig, kBandLittle, kBandAll, kBandAll, kBandAll})
          .AddSmoothingFactor(0.1)
          .AddProfileDataPath("band/test/data/profile.json")
          .AddOnline(true)
          .AddNumWarmups(1)
          .AddNumRuns(1)
          .AddAllowWorkSteal(true)
          .AddAvailabilityCheckIntervalMs(30000)
          .AddScheduleWindowSize(10)
          .Build();

  auto engine = Engine::Create(config);
  EXPECT_TRUE(engine);

  Model model;
  EXPECT_EQ(model.FromPath(BackendType::TfLite, "band/test/data/add.tflite"), kBandOk);
  EXPECT_EQ(engine->RegisterModel(&model), kBandOk);

  int execution_count = 0;
  engine->SetOnEndRequest(
      [&execution_count](int job_id, BandStatus status) { execution_count++; });

  for (int worker_id = 0; worker_id < engine->GetNumWorkers(); worker_id++) {
    EXPECT_EQ(engine->RequestSync(model.GetId(), {worker_id, true, -1, -1}),
              kBandOk);
    EXPECT_EQ(execution_count, worker_id + 1);
    EXPECT_EQ(engine->RequestSync(model.GetId(), {worker_id, false, -1, -1}),
              kBandOk);
    EXPECT_EQ(execution_count, worker_id + 1);
  }
}
}  // namespace Band

int main(int argc, char** argv) {
#ifdef BAND_TFLITE
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
#endif  // BAND_TFLITE
  return 0;
}
