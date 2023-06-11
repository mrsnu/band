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
#include "band/buffer/buffer.h"
#include "band/buffer/common_operator.h"
#include "band/buffer/image_operator.h"
#include "band/buffer/image_processor.h"
#include "band/config_builder.h"
#include "band/engine.h"
#include "band/interface/model_executor.h"
#include "band/interface/tensor.h"
#include "band/model.h"
#include "band/tensor.h"
#include "band/test/image_util.h"

namespace band {
namespace test {
using namespace interface;
TEST(TFLiteBackend, BackendInvoke) {
  tfl::TfLiteModel bin_model(0);
  EXPECT_EQ(bin_model.FromPath("band/test/data/add.tflite"), absl::OkStatus());
  tfl::TfLiteModelExecutor model_executor(0, 0, DeviceFlag::kCPU);
  EXPECT_EQ(model_executor.PrepareSubgraph(&bin_model), absl::OkStatus());
  EXPECT_TRUE(
      model_executor.ExecuteSubgraph(model_executor.GetLargestSubgraphKey())
          .ok());
}

TEST(TFLiteBackend, ModelSpec) {
  tfl::TfLiteModel bin_model(0);
  EXPECT_EQ(bin_model.FromPath("band/test/data/add.tflite"), absl::OkStatus());

  tfl::TfLiteModelExecutor model_executor(0, 0, DeviceFlag::kCPU);
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
  IModel* bin_model = BackendFactory::CreateModel(BackendType::kTfLite, 0);
  EXPECT_EQ(bin_model->FromPath("band/test/data/add.tflite"), absl::OkStatus());

  IModelExecutor* model_executor = BackendFactory::CreateModelExecutor(
      BackendType::kTfLite, 0, 0, DeviceFlag::kCPU);
  EXPECT_EQ(model_executor->PrepareSubgraph(bin_model), absl::OkStatus());

  SubgraphKey key = model_executor->GetLargestSubgraphKey();

  EXPECT_EQ(model_executor->GetInputs(key).size(), 1);
  EXPECT_EQ(model_executor->GetOutputs(key).size(), 1);

  std::array<float, 2> input = {1.f, 3.f};
  memcpy(model_executor->GetTensorView(key, model_executor->GetInputs(key)[0])
             ->GetData(),
         input.data(), input.size() * sizeof(float));

  EXPECT_EQ(model_executor->ExecuteSubgraph(key), absl::OkStatus());

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
      b.AddPlannerLogPath("band/test/data/log.json")
          .AddSchedulers({SchedulerType::kRoundRobin})
          .AddMinimumSubgraphSize(7)
          .AddSubgraphPreparationType(
              SubgraphPreparationType::kMergeUnitSubgraph)
          .AddCPUMask(CPUMaskFlag::kAll)
          .AddPlannerCPUMask(CPUMaskFlag::kPrimary)
          .AddWorkers({DeviceFlag::kCPU, DeviceFlag::kCPU})
          .AddWorkerNumThreads({3, 4})
          .AddWorkerCPUMasks({CPUMaskFlag::kBig, CPUMaskFlag::kLittle})
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
      model.FromPath(BackendType::kTfLite, "band/test/data/add.tflite").ok());
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
      b.AddPlannerLogPath("band/test/data/log.json")
          .AddSchedulers({SchedulerType::kFixedWorkerGlobalQueue})
          .AddMinimumSubgraphSize(7)
          .AddSubgraphPreparationType(
              SubgraphPreparationType::kMergeUnitSubgraph)
          .AddCPUMask(CPUMaskFlag::kAll)
          .AddPlannerCPUMask(CPUMaskFlag::kPrimary)
          .AddWorkers({DeviceFlag::kCPU, DeviceFlag::kCPU})
          .AddWorkerNumThreads({3, 4})
          .AddWorkerCPUMasks({CPUMaskFlag::kBig, CPUMaskFlag::kLittle})
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
      model.FromPath(BackendType::kTfLite, "band/test/data/add.tflite").ok());
  EXPECT_EQ(engine->RegisterModel(&model), absl::OkStatus());

  EXPECT_GE(
      engine->GetProfiled(engine->GetLargestSubgraphKey(model.GetId(), 0)), 0);
  EXPECT_GE(
      engine->GetExpected(engine->GetLargestSubgraphKey(model.GetId(), 0)), 0);
}

TEST(TFLiteBackend, SimpleEngineInvokeAsync) {
  RuntimeConfigBuilder b;
  RuntimeConfig config =
      b.AddPlannerLogPath("band/test/data/log.json")
          .AddSchedulers({SchedulerType::kShortestExpectedLatency})
          .AddMinimumSubgraphSize(7)
          .AddSubgraphPreparationType(
              SubgraphPreparationType::kMergeUnitSubgraph)
          .AddCPUMask(CPUMaskFlag::kAll)
          .AddPlannerCPUMask(CPUMaskFlag::kPrimary)
          .AddWorkers({DeviceFlag::kCPU, DeviceFlag::kCPU})
          .AddWorkerNumThreads({3, 4})
          .AddWorkerCPUMasks({CPUMaskFlag::kBig, CPUMaskFlag::kLittle})
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
      model.FromPath(BackendType::kTfLite, "band/test/data/add.tflite").ok());
  EXPECT_EQ(engine->RegisterModel(&model), absl::OkStatus());

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
  EXPECT_EQ(engine->Wait(job_id, {output_tensor}), absl::OkStatus());
  EXPECT_EQ(reinterpret_cast<float*>(output_tensor->GetData())[0], 3.f);
  EXPECT_EQ(reinterpret_cast<float*>(output_tensor->GetData())[1], 9.f);
  EXPECT_EQ(execution_count, 1);

  delete input_tensor;
  delete output_tensor;
}  // namespace

TEST(TFLiteBackend, SimpleEngineInvokeSyncOnWorker) {
  RuntimeConfigBuilder b;
  RuntimeConfig config =
      b.AddPlannerLogPath("band/test/data/log.json")
          .AddSchedulers({SchedulerType::kFixedWorker})
          .AddMinimumSubgraphSize(7)
          .AddSubgraphPreparationType(
              SubgraphPreparationType::kMergeUnitSubgraph)
          .AddCPUMask(CPUMaskFlag::kAll)
          .AddPlannerCPUMask(CPUMaskFlag::kPrimary)
#ifdef __ANDROID__
          .AddWorkers({DeviceFlag::kCPU, DeviceFlag::kCPU, DeviceFlag::kDSP,
                       DeviceFlag::kNPU, DeviceFlag::kGPU})
          .AddWorkerNumThreads({3, 4, 1, 1, 1})
          .AddWorkerCPUMasks({CPUMaskFlag::kBig, CPUMaskFlag::kLittle,
                              CPUMaskFlag::kAll, CPUMaskFlag::kAll,
                              CPUMaskFlag::kAll})
#else
          .AddWorkers({DeviceFlag::kCPU, DeviceFlag::kCPU})
          .AddWorkerNumThreads({3, 4})
          .AddWorkerCPUMasks({CPUMaskFlag::kBig, CPUMaskFlag::kLittle})
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
      model.FromPath(BackendType::kTfLite, "band/test/data/add.tflite").ok());
  EXPECT_EQ(engine->RegisterModel(&model), absl::OkStatus());

  Tensor* input_tensor = engine->CreateTensor(
      model.GetId(), engine->GetInputTensorIndices(model.GetId())[0]);
  Tensor* output_tensor = engine->CreateTensor(
      model.GetId(), engine->GetOutputTensorIndices(model.GetId())[0]);

  EXPECT_TRUE(input_tensor && output_tensor);

  std::array<float, 2> input = {1.f, 3.f};
  memcpy(input_tensor->GetData(), input.data(), input.size() * sizeof(float));

  std::cout << "Num workers " << engine->GetNumWorkers() << std::endl;
  for (size_t worker_id = 0; worker_id < engine->GetNumWorkers(); worker_id++) {
    std::cout << "Run on worker (device: "
              << ToString(engine->GetWorkerDevice(worker_id)) << ")"
              << std::endl;
    EXPECT_TRUE(engine
                    ->RequestSync(model.GetId(),
                                  {static_cast<int>(worker_id), true, -1, -1},
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
      b.AddPlannerLogPath("band/test/data/log.json")
          .AddSchedulers({SchedulerType::kFixedWorker})
          .AddMinimumSubgraphSize(7)
          .AddSubgraphPreparationType(
              SubgraphPreparationType::kMergeUnitSubgraph)
          .AddCPUMask(CPUMaskFlag::kAll)
          .AddPlannerCPUMask(CPUMaskFlag::kPrimary)
#ifdef __ANDROID__
          .AddWorkers({DeviceFlag::kCPU, DeviceFlag::kCPU, DeviceFlag::kDSP,
                       DeviceFlag::kNPU, DeviceFlag::kGPU})
          .AddWorkerNumThreads({3, 4, 1, 1, 1})
          .AddWorkerCPUMasks({CPUMaskFlag::kBig, CPUMaskFlag::kLittle,
                              CPUMaskFlag::kAll, CPUMaskFlag::kAll,
                              CPUMaskFlag::kAll})
#else
          .AddWorkers({DeviceFlag::kCPU, DeviceFlag::kCPU})
          .AddWorkerNumThreads({3, 4})
          .AddWorkerCPUMasks({CPUMaskFlag::kBig, CPUMaskFlag::kLittle})
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
      model.FromPath(BackendType::kTfLite, "band/test/data/add.tflite").ok());
  EXPECT_EQ(engine->RegisterModel(&model), absl::OkStatus());

  int execution_count = 0;
  engine->SetOnEndRequest([&execution_count](int job_id, absl::Status status) {
    execution_count++;
  });

  for (size_t worker_id = 0; worker_id < engine->GetNumWorkers(); worker_id++) {
    EXPECT_TRUE(engine
                    ->RequestSync(model.GetId(),
                                  {static_cast<int>(worker_id), true, -1, -1})
                    .ok());
    EXPECT_EQ(execution_count, worker_id + 1);
    EXPECT_TRUE(engine
                    ->RequestSync(model.GetId(),
                                  {static_cast<int>(worker_id), false, -1, -1})
                    .ok());
    EXPECT_EQ(execution_count, worker_id + 1);
  }
}

TEST(TFLiteBackend, ClassificationQuantTest) {
  RuntimeConfigBuilder b;
  RuntimeConfig config =
      b.AddPlannerLogPath("band/test/data/log.json")
          .AddSchedulers({SchedulerType::kFixedWorker})
          .AddMinimumSubgraphSize(7)
          .AddSubgraphPreparationType(
              SubgraphPreparationType::kMergeUnitSubgraph)
          .AddCPUMask(CPUMaskFlag::kAll)
          .AddPlannerCPUMask(CPUMaskFlag::kPrimary)
#ifdef __ANDROID__
          .AddWorkers({DeviceFlag::kCPU, DeviceFlag::kCPU, DeviceFlag::kDSP,
                       DeviceFlag::kNPU, DeviceFlag::kGPU})
          .AddWorkerNumThreads({3, 4, 1, 1, 1})
          .AddWorkerCPUMasks({CPUMaskFlag::kBig, CPUMaskFlag::kLittle,
                              CPUMaskFlag::kAll, CPUMaskFlag::kAll,
                              CPUMaskFlag::kAll})
#else
          .AddWorkers({DeviceFlag::kCPU, DeviceFlag::kCPU})
          .AddWorkerNumThreads({3, 4})
          .AddWorkerCPUMasks({CPUMaskFlag::kBig, CPUMaskFlag::kLittle})
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
  std::shared_ptr<Buffer> image_buffer = LoadImage("band/test/data/cat.jpg");

  Model model;
  EXPECT_TRUE(model
                  .FromPath(BackendType::kTfLite,
                            "band/test/data/mobilenet_v2_1.0_224_quant.tflite")
                  .ok());
  EXPECT_EQ(engine->RegisterModel(&model), absl::OkStatus());

  Tensor* input_tensor = engine->CreateTensor(
      model.GetId(), engine->GetInputTensorIndices(model.GetId())[0]);
  std::shared_ptr<Buffer> tensor_buffer(Buffer::CreateFromTensor(input_tensor));

  ImageProcessorBuilder preprocessor_builder;
  // by default, the image is resized to input size
  absl::StatusOr<std::unique_ptr<BufferProcessor>> preprocessor =
      preprocessor_builder.Build();
  EXPECT_TRUE(preprocessor.ok());
  EXPECT_TRUE(
      preprocessor.value()->Process(*image_buffer, *tensor_buffer).ok());
  // confirm that the image is resized to 224x224 and converted to RGB
  Tensor* output_tensor = engine->CreateTensor(
      model.GetId(), engine->GetOutputTensorIndices(model.GetId())[0]);
  EXPECT_TRUE(engine
                  ->RequestSync(model.GetId(), {0, false, -1, -1},
                                {input_tensor}, {output_tensor})
                  .ok());

  // TODO: postprocessing library
  std::vector<unsigned char> output_data;
  output_data.resize(output_tensor->GetNumElements());
  memcpy(output_data.data(), output_tensor->GetData(),
         output_tensor->GetNumElements() * sizeof(unsigned char));

  size_t max_index = 0;
  unsigned char max_value = 0;
  for (size_t i = 0; i < output_data.size(); ++i) {
    if (output_data[i] > max_value) {
      max_value = output_data[i];
      max_index = i;
    }
  }
  // tiger cat
  EXPECT_EQ(max_index, 282);
}

TEST(TFLiteBackend, ClassificationTest) {
  RuntimeConfigBuilder b;
  RuntimeConfig config =
      b.AddPlannerLogPath("band/test/data/log.json")
          .AddSchedulers({SchedulerType::kFixedWorker})
          .AddMinimumSubgraphSize(7)
          .AddSubgraphPreparationType(
              SubgraphPreparationType::kMergeUnitSubgraph)
          .AddCPUMask(CPUMaskFlag::kAll)
          .AddPlannerCPUMask(CPUMaskFlag::kPrimary)
#ifdef __ANDROID__
          .AddWorkers({DeviceFlag::kCPU, DeviceFlag::kCPU, DeviceFlag::kDSP,
                       DeviceFlag::kNPU, DeviceFlag::kGPU})
          .AddWorkerNumThreads({3, 4, 1, 1, 1})
          .AddWorkerCPUMasks({CPUMaskFlag::kBig, CPUMaskFlag::kLittle,
                              CPUMaskFlag::kAll, CPUMaskFlag::kAll,
                              CPUMaskFlag::kAll})
#else
          .AddWorkers({DeviceFlag::kCPU, DeviceFlag::kCPU})
          .AddWorkerNumThreads({3, 4})
          .AddWorkerCPUMasks({CPUMaskFlag::kBig, CPUMaskFlag::kLittle})
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
  std::shared_ptr<Buffer> image_buffer = LoadImage("band/test/data/cat.jpg");

  Model model;
  EXPECT_TRUE(
      model
          .FromPath(
              BackendType::kTfLite,
              "band/test/data/lite-model_mobilenet_v2_100_224_fp32_1.tflite")
          .ok());
  EXPECT_EQ(engine->RegisterModel(&model), absl::OkStatus());

  Tensor* input_tensor = engine->CreateTensor(
      model.GetId(), engine->GetInputTensorIndices(model.GetId())[0]);
  std::shared_ptr<Buffer> tensor_buffer(Buffer::CreateFromTensor(input_tensor));
  // image -> rgb -> normalize
  ImageProcessorBuilder preprocessor_builder;
  preprocessor_builder
      .AddOperation(std::make_unique<buffer::Resize>(224, 224))
      // TODO: add data type conversion
      .AddOperation(std::make_unique<buffer::Normalize>(127.5f, 127.5f, false));
  // by default, the image is resized to input size
  absl::StatusOr<std::unique_ptr<BufferProcessor>> preprocessor =
      preprocessor_builder.Build();
  EXPECT_TRUE(preprocessor.ok());
  EXPECT_TRUE(
      preprocessor.value()->Process(*image_buffer, *tensor_buffer).ok());

  for (size_t i = 0; i < input_tensor->GetNumElements(); ++i) {
    EXPECT_GT(reinterpret_cast<float*>(input_tensor->GetData())[i], -1.0f);
    EXPECT_LT(reinterpret_cast<float*>(input_tensor->GetData())[i], 1.0f);
  }

  // confirm that the image is resized to 224x224 and converted to RGB
  Tensor* output_tensor = engine->CreateTensor(
      model.GetId(), engine->GetOutputTensorIndices(model.GetId())[0]);
  EXPECT_TRUE(engine
                  ->RequestSync(model.GetId(), {0, false, -1, -1},
                                {input_tensor}, {output_tensor})
                  .ok());

  // TODO: postprocessing library
  std::vector<unsigned char> output_data;
  output_data.resize(output_tensor->GetNumElements());
  memcpy(output_data.data(), output_tensor->GetData(),
         output_tensor->GetNumElements() * sizeof(unsigned char));

  size_t max_index = 0;
  unsigned char max_value = 0;
  for (size_t i = 0; i < output_data.size(); ++i) {
    if (output_data[i] > max_value) {
      max_value = output_data[i];
      max_index = i;
    }
  }
  // tiger cat
  EXPECT_EQ(max_index, 282);
}

}  // namespace test
}  // namespace band

int main(int argc, char** argv) {
#ifdef BAND_TFLITE
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
#endif  // BAND_TFLITE
  return 0;
}
