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
namespace test {
using namespace interface;

struct ModelPartitionTestsFixture
    : public testing::TestWithParam<
          std::tuple<std::string, SubgraphPreparationType>> {};

TEST_P(ModelPartitionTestsFixture, ModelPartitionTest) {
  std::string model_name = "band/test/data/" + std::get<0>(GetParam());
  SubgraphPreparationType subgraph_type = std::get<1>(GetParam());

  RuntimeConfigBuilder b;
  RuntimeConfig config =
      b.AddPlannerLogPath("band/test/data/log.json")
          .AddSchedulers({SchedulerType::kLeastSlackTimeFirst})
          .AddMinimumSubgraphSize(7)
          .AddSubgraphPreparationType(subgraph_type)
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
          .AddLatencySmoothingFactor(0.1)
          .AddProfilePath("band/test/data/profile.json")
          .AddNumWarmups(1)
          .AddNumRuns(1)
          .AddAvailabilityCheckIntervalMs(30000)
          .AddScheduleWindowSize(10)
          .Build();

  auto engine = Engine::Create(config);
  EXPECT_NE(engine, nullptr);

  Model model;
  EXPECT_EQ(model.FromPath(BackendType::kTfLite, model_name.c_str()),
            absl::OkStatus());
  EXPECT_EQ(engine->RegisterModel(&model), absl::OkStatus());
}

INSTANTIATE_TEST_SUITE_P(
    ModelPartitionTests, ModelPartitionTestsFixture,
    testing::Values(
        std::make_tuple("retinaface-mbv2-int8.tflite",
                        SubgraphPreparationType::kUnitSubgraph),
        std::make_tuple("lite-model_efficientdet_lite0_int8_1.tflite",
                        SubgraphPreparationType::kMergeUnitSubgraph),
        std::make_tuple("lite-model_efficientdet_lite0_int8_1.tflite",
                        SubgraphPreparationType::kUnitSubgraph),
        std::make_tuple("ICN_quant.tflite",
                        SubgraphPreparationType::kMergeUnitSubgraph),
        std::make_tuple(
            "magenta_arbitrary-image-stylization-v1-256_int8_transfer_1.tflite",
            SubgraphPreparationType::kMergeUnitSubgraph),
        std::make_tuple("retinaface_mbv2_quant_160.tflite",
                        SubgraphPreparationType::kMergeUnitSubgraph)));
}  // namespace test
}  // namespace band

int main(int argc, char** argv) {
#ifdef BAND_TFLITE
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
#endif  // BAND_TFLITE
  return 0;
}