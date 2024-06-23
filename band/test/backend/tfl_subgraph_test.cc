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
#ifdef CL_DELEGATE_NO_GL
          .AddWorkers({DeviceFlag::kCPU, DeviceFlag::kCPU, DeviceFlag::kGPU})
          .AddWorkerNumThreads({3, 4, 1})
          .AddWorkerCPUMasks({CPUMaskFlag::kBig, CPUMaskFlag::kLittle,
                              CPUMaskFlag::kAll})
#elif __ANDROID__
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
#endif  // CL_DELEGATE_NO_GL & __ANDROID__
          .AddSmoothingFactor(0.1)
          .AddProfileDataPath("band/test/data/profile.json")
          .AddOnline(true)
          .AddNumWarmups(1)
          .AddNumRuns(1)
          .AddAllowWorkSteal(true)
          .AddAvailabilityCheckIntervalMs(30000)
          .AddScheduleWindowSize(10)
          .Build()
          .value();

  auto engine = Engine::Create(config);
  EXPECT_TRUE(engine);

  Model model;
  EXPECT_EQ(model.FromPath(BackendType::kTfLite, model_name.c_str()),
            absl::OkStatus());
  EXPECT_EQ(engine->RegisterModel(&model), absl::OkStatus());
}

INSTANTIATE_TEST_SUITE_P(
    ModelPartitionTests, ModelPartitionTestsFixture,
    testing::Values(
        std::make_tuple("lite-model_efficientdet_lite0_int8_1.tflite",
                        SubgraphPreparationType::kMergeUnitSubgraph),
        std::make_tuple("lite-model_efficientdet_lite0_int8_1.tflite",
                        SubgraphPreparationType::kFallbackPerWorker),
        std::make_tuple("ICN_quant.tflite",
                        SubgraphPreparationType::kMergeUnitSubgraph),
        std::make_tuple("retinaface_mbv2_quant_160.tflite",
                        SubgraphPreparationType::kMergeUnitSubgraph),
        std::make_tuple("ffnet_40s_quantized.tflite",
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