#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "band/common.h"
#include "band/config.h"
#include "band/config_builder.h"
#include "band/engine.h"
#include "band/model_analyzer.h"

namespace band {
namespace test {

TEST(ModelAnalyzerTest, CreateSubgraphsTest) {
  RuntimeConfigBuilder b;
  RuntimeConfig config =
      b.AddPlannerLogPath("band/test/data/log.json")
          .AddSchedulers({SchedulerType::kHeterogeneousEarliestFinishTime})
          .AddMinimumSubgraphSize(1)
          .AddSubgraphPreparationType(
              SubgraphPreparationType::kNoFallbackSubgraph)
#ifdef __ANDROID__
          .AddCPUMask(CPUMaskFlag::kBig)
          .AddPlannerCPUMask(CPUMaskFlag::kBig)
          .AddWorkers({DeviceFlag::kCPU, DeviceFlag::kDSP, DeviceFlag::kNPU,
                       DeviceFlag::kGPU})
          .AddWorkerNumThreads({4, 1, 1, 1})
          .AddWorkerCPUMasks({CPUMaskFlag::kPrimary, CPUMaskFlag::kBig,
                              CPUMaskFlag::kBig, CPUMaskFlag::kBig})
#else
          .AddCPUMask(CPUMaskFlag::kBig)
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
  std::unique_ptr<Engine> engine = Engine::Create(config);
  EXPECT_NE(engine, nullptr);

  Model model;
  ModelAnalyzer analyzer(*engine, true, config.subgraph_config, /*model*/nullptr,
                         BackendType::kTfLite);
}

}  // namespace test
}  // namespace band

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}