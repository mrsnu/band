#include "band/config_builder.h"

#include <gtest/gtest.h>

namespace band {
namespace test {

TEST(ConfigBuilderTest, LatencyProfileConfigBuilderTest) {
  LatencyProfileConfigBuilder b;
  LatencyProfileConfig config_ok = b.AddSmoothingFactor(0.1f).Build();

  EXPECT_EQ(config_ok.smoothing_factor, 0.1f);

  b.AddSmoothingFactor(-1.0f);
  EXPECT_FALSE(b.IsValid());
}

TEST(ConfigBuilderTest, PlannerConfigBuilderTest) {
  PlannerConfigBuilder b;
  PlannerConfig config_ok = b.AddLogPath("band/test/data/config.json")
                                .AddScheduleWindowSize(5)
                                .AddSchedulers({SchedulerType::kFixedWorker})
                                .Build();
  EXPECT_EQ(config_ok.log_path, "band/test/data/config.json");
  EXPECT_EQ(config_ok.schedule_window_size, 5);
  EXPECT_EQ(config_ok.cpu_mask, CPUMaskFlag::kAll);

  b.AddScheduleWindowSize(-1);
  EXPECT_FALSE(b.IsValid());
}

TEST(ConfigBuilderTest, WorkerConfigBuilderTest) {
  WorkerConfigBuilder b;
  WorkerConfig config_ok =
      b.AddAvailabilityCheckIntervalMs(1000)
          .AddWorkers({DeviceFlag::kCPU, DeviceFlag::kDSP})
          .AddCPUMasks({CPUMaskFlag::kAll, CPUMaskFlag::kAll})
          .AddNumThreads({1, 1})
          .Build();
  EXPECT_EQ(config_ok.availability_check_interval_ms, 1000);
  EXPECT_EQ(config_ok.workers.size(), 2);
  EXPECT_EQ(config_ok.cpu_masks.size(), config_ok.workers.size());
  EXPECT_EQ(config_ok.num_threads.size(), config_ok.workers.size());

  b.AddWorkers({DeviceFlag::kCPU});
  EXPECT_FALSE(b.IsValid());
  b.AddWorkers({DeviceFlag::kCPU, DeviceFlag::kGPU});
  EXPECT_TRUE(b.IsValid());
}

TEST(ConfigBuilderTest, RuntimeConfigBuilderTest) {
  RuntimeConfigBuilder b;
  RuntimeConfig config_ok =
      b.AddNumWarmups(1)
          .AddNumRuns(1)
          .AddLatencySmoothingFactor(0.1)
          .AddProfilePath("band/test/data/config.json")
          .AddMinimumSubgraphSize(5)
          .AddSubgraphPreparationType(
              SubgraphPreparationType::kNoFallbackSubgraph)
          .AddPlannerLogPath("band/test/data/config.json")
          .AddScheduleWindowSize(1)
          .AddSchedulers({SchedulerType::kFixedWorker})
          .AddPlannerCPUMask(CPUMaskFlag::kBig)
          .AddWorkers({})
          .AddWorkerCPUMasks({})
          .AddWorkerNumThreads({})
          .AddAvailabilityCheckIntervalMs(100)
          .AddCPUMask(CPUMaskFlag::kPrimary)
          .Build();
  EXPECT_EQ(config_ok.profile_config.num_warmups, 1);
  EXPECT_EQ(config_ok.profile_config.num_runs, 1);
  EXPECT_EQ(config_ok.profile_config.latency_config.smoothing_factor, 0.1f);
  EXPECT_EQ(config_ok.profile_config.profile_path,
            "band/test/data/config.json");
  EXPECT_EQ(config_ok.subgraph_config.minimum_subgraph_size, 5);
  EXPECT_EQ(config_ok.subgraph_config.subgraph_preparation_type,
            SubgraphPreparationType::kNoFallbackSubgraph);
  EXPECT_EQ(config_ok.cpu_mask, CPUMaskFlag::kPrimary);
  EXPECT_EQ(config_ok.planner_config.log_path, "band/test/data/config.json");
  EXPECT_EQ(config_ok.planner_config.schedule_window_size, 1);
  EXPECT_EQ(config_ok.planner_config.schedulers[0],
            SchedulerType::kFixedWorker);
  EXPECT_EQ(config_ok.planner_config.cpu_mask, CPUMaskFlag::kBig);
  EXPECT_EQ(config_ok.worker_config.workers[0], DeviceFlag::kCPU);
  EXPECT_EQ(config_ok.worker_config.cpu_masks[0], CPUMaskFlag::kAll);
  EXPECT_EQ(config_ok.worker_config.num_threads[0], 1);
  EXPECT_EQ(config_ok.worker_config.availability_check_interval_ms, 100);
}

TEST(ConfigBuilderTest, DefaultValueTest) {
  RuntimeConfigBuilder b;
  RuntimeConfig config_ok =
      b.AddSchedulers({SchedulerType::kFixedWorker}).Build();
  EXPECT_EQ(config_ok.profile_config.num_warmups, 1);
  EXPECT_EQ(config_ok.profile_config.num_runs, 1);
  EXPECT_EQ(config_ok.profile_config.profile_path, "");
  EXPECT_EQ(config_ok.profile_config.latency_config.smoothing_factor, 0.1f);
  EXPECT_EQ(config_ok.planner_config.log_path, "");
  EXPECT_EQ(config_ok.planner_config.schedulers[0],
            SchedulerType::kFixedWorker);
  EXPECT_EQ(config_ok.planner_config.schedule_window_size, INT_MAX);
  EXPECT_EQ(config_ok.planner_config.cpu_mask, CPUMaskFlag::kAll);
  EXPECT_EQ(config_ok.worker_config.workers[0], DeviceFlag::kCPU);
  EXPECT_EQ(config_ok.worker_config.workers[1], DeviceFlag::kGPU);
  EXPECT_EQ(config_ok.worker_config.workers[2], DeviceFlag::kDSP);
  EXPECT_EQ(config_ok.worker_config.workers[3], DeviceFlag::kNPU);
  EXPECT_EQ(config_ok.worker_config.cpu_masks[0], CPUMaskFlag::kAll);
  EXPECT_EQ(config_ok.worker_config.cpu_masks[1], CPUMaskFlag::kAll);
  EXPECT_EQ(config_ok.worker_config.cpu_masks[2], CPUMaskFlag::kAll);
  EXPECT_EQ(config_ok.worker_config.cpu_masks[3], CPUMaskFlag::kAll);
  EXPECT_EQ(config_ok.worker_config.num_threads[0], 1);
  EXPECT_EQ(config_ok.worker_config.num_threads[1], 1);
  EXPECT_EQ(config_ok.worker_config.num_threads[2], 1);
  EXPECT_EQ(config_ok.worker_config.num_threads[3], 1);
  EXPECT_EQ(config_ok.worker_config.availability_check_interval_ms, 30000);
  EXPECT_EQ(config_ok.subgraph_config.minimum_subgraph_size, 7);
  EXPECT_EQ(config_ok.subgraph_config.subgraph_preparation_type,
            SubgraphPreparationType::kMergeUnitSubgraph);
  EXPECT_EQ(config_ok.cpu_mask, CPUMaskFlag::kAll);
}

}  // namespace test
}  // namespace band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}