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

namespace band {
using namespace interface;
TEST(TFLiteBackend, BackendInvoke) {
  tfl::TfLiteModel bin_model(0);
  bin_model.FromPath("band/test/data/add.tflite");

  tfl::TfLiteModelExecutor model_executor(0, 0, DeviceFlags::CPU);
  EXPECT_TRUE(model_executor.PrepareSubgraph(&bin_model).ok());
  EXPECT_TRUE(
      model_executor.ExecuteSubgraph(model_executor.GetLargestSubgraphKey())
          .ok());
}

TEST(TFLiteBackend, ModelSpec) {
  tfl::TfLiteModel bin_model(0);
  bin_model.FromPath("band/test/data/add.tflite");

  tfl::TfLiteModelExecutor model_executor(0, 0, DeviceFlags::CPU);
  ModelSpec model_spec =
      model_executor.InvestigateModelSpec(&bin_model).value();

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

  IModelExecutor* model_executor = BackendFactory::CreateModelExecutor(
      BackendType::TfLite, 0, 0, DeviceFlags::CPU);
  EXPECT_TRUE(model_executor->PrepareSubgraph(bin_model).ok());

  SubgraphKey key = model_executor->GetLargestSubgraphKey();

  EXPECT_EQ(model_executor->GetInputs(key).size(), 1);
  EXPECT_EQ(model_executor->GetOutputs(key).size(), 1);

  std::array<float, 2> input = {1.f, 3.f};
  memcpy(model_executor->GetTensorView(key, model_executor->GetInputs(key)[0])
             ->GetData(),
         input.data(), input.size() * sizeof(float));

  EXPECT_TRUE(model_executor->ExecuteSubgraph(key).ok());

  auto output_tensor =
      model_executor->GetTensorView(key, model_executor->GetOutputs(key)[0]);
  EXPECT_EQ(reinterpret_cast<float*>(output_tensor->GetData())[0], 3.f);
  EXPECT_EQ(reinterpret_cast<float*>(output_tensor->GetData())[1], 9.f);

  delete bin_model;
  delete model_executor;
}

TEST(TFLiteBackend, SimpleEngineInvokeSync) {
  RuntimeConfigBuilder b;
  RuntimeConfig config =
      b.AddPlannerLogPath("band/test/data/log.csv")
          .AddSchedulers({SchedulerType::RoundRobin})
          .AddMinimumSubgraphSize(7)
          .AddSubgraphPreparationType(
              SubgraphPreparationType::MergeUnitSubgraph)
          .AddCPUMask(CPUMaskFlags::All)
          .AddPlannerCPUMask(CPUMaskFlags::Primary)
          .AddWorkers({DeviceFlags::CPU, DeviceFlags::CPU})
          .AddWorkerNumThreads({3, 4})
          .AddWorkerCPUMasks({CPUMaskFlags::Big, CPUMaskFlags::Little})
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
  EXPECT_EQ(
      model.FromPath(BackendType::TfLite, "band/test/data/add.tflite"), absl::OkStatus());
  EXPECT_EQ(engine->RegisterModel(&model), absl::OkStatus());

  Tensor* input_tensor = engine->CreateTensor(
      model.GetId(), engine->GetInputTensorIndices(model.GetId())[0]);
  Tensor* output_tensor = engine->CreateTensor(
      model.GetId(), engine->GetOutputTensorIndices(model.GetId())[0]);

  EXPECT_TRUE(input_tensor && output_tensor);

  int execution_count = 0;
  engine->SetOnEndRequest(
      [&execution_count](int, absl::Status) { execution_count++; });

  std::array<float, 2> input = {1.f, 3.f};
  memcpy(input_tensor->GetData(), input.data(), input.size() * sizeof(float));

  EXPECT_TRUE(engine
                  ->RequestSync(model.GetId(),
                                RequestOption::GetDefaultOption(),
                                {input_tensor}, {output_tensor})
                  .ok());
  EXPECT_EQ(reinterpret_cast<float*>(output_tensor->GetData())[0], 3.f);
  EXPECT_EQ(reinterpret_cast<float*>(output_tensor->GetData())[1], 9.f);
  EXPECT_EQ(execution_count, 1);

  delete input_tensor;
  delete output_tensor;
}

TEST(TFLiteBackend, SimpleEngineProfile) {
  RuntimeConfigBuilder b;
  RuntimeConfig config =
      b.AddPlannerLogPath("band/test/data/log.csv")
          .AddSchedulers({SchedulerType::FixedWorkerGlobalQueue})
          .AddMinimumSubgraphSize(7)
          .AddSubgraphPreparationType(
              SubgraphPreparationType::MergeUnitSubgraph)
          .AddCPUMask(CPUMaskFlags::All)
          .AddPlannerCPUMask(CPUMaskFlags::Primary)
          .AddWorkers({DeviceFlags::CPU, DeviceFlags::CPU})
          .AddWorkerNumThreads({3, 4})
          .AddWorkerCPUMasks({CPUMaskFlags::Big, CPUMaskFlags::Little})
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
  EXPECT_TRUE(
      model.FromPath(BackendType::TfLite, "band/test/data/add.tflite").ok());
  EXPECT_TRUE(engine->RegisterModel(&model).ok());

  EXPECT_GT(
      engine->GetProfiled(engine->GetLargestSubgraphKey(model.GetId(), 0)), 0);
  EXPECT_GT(
      engine->GetExpected(engine->GetLargestSubgraphKey(model.GetId(), 0)), 0);
}

TEST(TFLiteBackend, SimpleEngineInvokeAsync) {
  RuntimeConfigBuilder b;
  RuntimeConfig config =
      b.AddPlannerLogPath("band/test/data/log.csv")
          .AddSchedulers({SchedulerType::ShortestExpectedLatency})
          .AddMinimumSubgraphSize(7)
          .AddSubgraphPreparationType(
              SubgraphPreparationType::MergeUnitSubgraph)
          .AddCPUMask(CPUMaskFlags::All)
          .AddPlannerCPUMask(CPUMaskFlags::Primary)
          .AddWorkers({DeviceFlags::CPU, DeviceFlags::CPU})
          .AddWorkerNumThreads({3, 4})
          .AddWorkerCPUMasks({CPUMaskFlags::Big, CPUMaskFlags::Little})
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
  EXPECT_TRUE(
      model.FromPath(BackendType::TfLite, "band/test/data/add.tflite").ok());
  EXPECT_TRUE(engine->RegisterModel(&model).ok());

  Tensor* input_tensor = engine->CreateTensor(
      model.GetId(), engine->GetInputTensorIndices(model.GetId())[0]);
  Tensor* output_tensor = engine->CreateTensor(
      model.GetId(), engine->GetOutputTensorIndices(model.GetId())[0]);

  EXPECT_TRUE(input_tensor && output_tensor);

  std::array<float, 2> input = {1.f, 3.f};
  memcpy(input_tensor->GetData(), input.data(), input.size() * sizeof(float));

  int execution_count = 0;
  engine->SetOnEndRequest(
      [&execution_count](int, absl::Status) { execution_count++; });

  auto job_id =
      engine
          ->RequestAsync(model.GetId(), RequestOption::GetDefaultOption(),
                         {input_tensor})
          .value();
  EXPECT_TRUE(engine->Wait(job_id, {output_tensor}).ok());
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
          .AddSchedulers({SchedulerType::FixedWorker})
          .AddMinimumSubgraphSize(7)
          .AddSubgraphPreparationType(
              SubgraphPreparationType::MergeUnitSubgraph)
          .AddCPUMask(CPUMaskFlags::All)
          .AddPlannerCPUMask(CPUMaskFlags::Primary)
#ifdef __ANDROID__
          .AddWorkers({DeviceFlags::CPU, DeviceFlags::CPU, DeviceFlags::DSP,
                       DeviceFlags::NPU, DeviceFlags::GPU})
          .AddWorkerNumThreads({3, 4, 1, 1, 1})
          .AddWorkerCPUMasks({CPUMaskFlags::Big, CPUMaskFlags::Little,
                              CPUMaskFlags::All, CPUMaskFlags::All,
                              CPUMaskFlags::All})
#else
          .AddWorkers({DeviceFlags::CPU, DeviceFlags::CPU})
          .AddWorkerNumThreads({3, 4})
          .AddWorkerCPUMasks({CPUMaskFlags::Big, CPUMaskFlags::Little})
#endif  // __ANDROID__
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
  EXPECT_TRUE(
      model.FromPath(BackendType::TfLite, "band/test/data/add.tflite").ok());
  EXPECT_TRUE(engine->RegisterModel(&model).ok());

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
              << GetName(engine->GetWorkerDevice(worker_id)) << ")"
              << std::endl;
    EXPECT_TRUE(engine
                    ->RequestSync(model.GetId(), {worker_id, true, -1, -1},
                                  {input_tensor}, {output_tensor})
                    .ok());
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
          .AddSchedulers({SchedulerType::FixedWorker})
          .AddMinimumSubgraphSize(7)
          .AddSubgraphPreparationType(
              SubgraphPreparationType::MergeUnitSubgraph)
          .AddCPUMask(CPUMaskFlags::All)
          .AddPlannerCPUMask(CPUMaskFlags::Primary)
#ifdef __ANDROID__
          .AddWorkers({DeviceFlags::CPU, DeviceFlags::CPU, DeviceFlags::DSP,
                       DeviceFlags::NPU, DeviceFlags::GPU})
          .AddWorkerNumThreads({3, 4, 1, 1, 1})
          .AddWorkerCPUMasks({CPUMaskFlags::Big, CPUMaskFlags::Little,
                              CPUMaskFlags::All, CPUMaskFlags::All,
                              CPUMaskFlags::All})
#else
          .AddWorkers({DeviceFlags::CPU, DeviceFlags::CPU})
          .AddWorkerNumThreads({3, 4})
          .AddWorkerCPUMasks({CPUMaskFlags::Big, CPUMaskFlags::Little})
#endif  // __ANDROID__
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
  EXPECT_TRUE(
      model.FromPath(BackendType::TfLite, "band/test/data/add.tflite").ok());
  EXPECT_TRUE(engine->RegisterModel(&model).ok());

  int execution_count = 0;
  engine->SetOnEndRequest([&execution_count](int job_id, absl::Status status) {
    execution_count++;
  });

  for (int worker_id = 0; worker_id < engine->GetNumWorkers(); worker_id++) {
    EXPECT_TRUE(
        engine->RequestSync(model.GetId(), {worker_id, true, -1, -1}).ok());
    EXPECT_EQ(execution_count, worker_id + 1);
    EXPECT_TRUE(
        engine->RequestSync(model.GetId(), {worker_id, false, -1, -1}).ok());
    EXPECT_EQ(execution_count, worker_id + 1);
  }
}
}  // namespace band

int main(int argc, char** argv) {
#ifdef BAND_TFLITE
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
#endif  // BAND_TFLITE
  return 0;
}
