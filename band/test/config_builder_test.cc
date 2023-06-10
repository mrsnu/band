#include "band/config_builder.h"

#include <gtest/gtest.h>

namespace band {
namespace test {

TEST(ConfigBuilderTest, ProfileConfigBuilderTest) {
  ProfileConfigBuilder b;
  ProfileConfig config_ok = b.AddOnline(false)
                                .AddNumRuns(3)
                                .AddNumWarmups(3)
                                .AddProfileDataPath("hello")
                                .Build();

  EXPECT_EQ(config_ok.online, false);
  EXPECT_EQ(config_ok.num_runs, 3);
  EXPECT_EQ(config_ok.num_warmups, 3);

  b.AddNumRuns(-1);
  EXPECT_FALSE(b.IsValid());
  b.AddNumRuns(1);
  b.AddOnline(true);
  EXPECT_TRUE(b.IsValid());
}

TEST(ConfigBuilderTest, PlannerConfigBuilderTest) {
  PlannerConfigBuilder b;
  PlannerConfig config_ok = b.AddLogPath("band/test/data/config.json")
                                .AddScheduleWindowSize(5)
                                .AddSchedulers({SchedulerType::kBandFixedWorker})
                                .Build();
  EXPECT_EQ(config_ok.log_path, "band/test/data/config.json");
  EXPECT_EQ(config_ok.schedule_window_size, 5);
  EXPECT_EQ(config_ok.cpu_mask, CPUMaskFlag::kBandAll);

  b.AddScheduleWindowSize(-1);
  EXPECT_FALSE(b.IsValid());
}

TEST(ConfigBuilderTest, WorkerConfigBuilderTest) {
  WorkerConfigBuilder b;
  WorkerConfig config_ok = b.AddAllowWorkSteal(false)
                               .AddAvailabilityCheckIntervalMs(1000)
                               .AddWorkers({DeviceFlag::kBandCPU, DeviceFlag::kBandDSP})
                               .AddCPUMasks({CPUMaskFlag::kBandAll, CPUMaskFlag::kBandAll})
                               .AddNumThreads({1, 1})
                               .Build();
  EXPECT_EQ(config_ok.allow_worksteal, false);
  EXPECT_EQ(config_ok.availability_check_interval_ms, 1000);
  EXPECT_EQ(config_ok.workers.size(), 2);
  EXPECT_EQ(config_ok.cpu_masks.size(), config_ok.workers.size());
  EXPECT_EQ(config_ok.num_threads.size(), config_ok.workers.size());

  b.AddWorkers({DeviceFlag::kBandCPU});
  EXPECT_FALSE(b.IsValid());
  b.AddWorkers({DeviceFlag::kBandCPU, DeviceFlag::kBandGPU});
  EXPECT_TRUE(b.IsValid());
}

TEST(ConfigBuilderTest, RuntimeConfigBuilderTest) {
  RuntimeConfigBuilder b;
  RuntimeConfig config_ok =
      b.AddOnline(true)
          .AddNumWarmups(1)
          .AddNumRuns(1)
          .AddCopyComputationRatio({1, 2, 3, 4})
          .AddSmoothingFactor(0.1)
          .AddProfileDataPath("band/test/data/config.json")
          .AddMinimumSubgraphSize(5)
          .AddSubgraphPreparationType(SubgraphPreparationType::kBandMergeUnitSubgraph)
          .AddPlannerLogPath("band/test/data/config.json")
          .AddScheduleWindowSize(1)
          .AddSchedulers({SchedulerType::kBandFixedWorker})
          .AddPlannerCPUMask(CPUMaskFlag::kBandBig)
          .AddWorkers({})
          .AddWorkerCPUMasks({})
          .AddWorkerNumThreads({})
          .AddAllowWorkSteal(true)
          .AddAvailabilityCheckIntervalMs(100)
          .AddCPUMask(CPUMaskFlag::kBandPrimary)
          .Build();
  EXPECT_EQ(config_ok.profile_config.online, true);
  EXPECT_EQ(config_ok.profile_config.num_warmups, 1);
  EXPECT_EQ(config_ok.profile_config.num_runs, 1);
  EXPECT_EQ(config_ok.profile_config.copy_computation_ratio[0], 1);
  EXPECT_EQ(config_ok.profile_config.copy_computation_ratio[1], 2);
  EXPECT_EQ(config_ok.profile_config.copy_computation_ratio[2], 3);
  EXPECT_EQ(config_ok.profile_config.copy_computation_ratio[3], 4);
  EXPECT_EQ(config_ok.profile_config.smoothing_factor, 0.1f);
  EXPECT_EQ(config_ok.profile_config.profile_data_path,
            "band/test/data/config.json");
  EXPECT_EQ(config_ok.subgraph_config.minimum_subgraph_size, 5);
  EXPECT_EQ(config_ok.subgraph_config.subgraph_preparation_type,
            SubgraphPreparationType::kBandMergeUnitSubgraph);
  EXPECT_EQ(config_ok.cpu_mask, CPUMaskFlag::kBandPrimary);
  EXPECT_EQ(config_ok.planner_config.log_path, "band/test/data/config.json");
  EXPECT_EQ(config_ok.planner_config.schedule_window_size, 1);
  EXPECT_EQ(config_ok.planner_config.schedulers[0], SchedulerType::kBandFixedWorker);
  EXPECT_EQ(config_ok.planner_config.cpu_mask, CPUMaskFlag::kBandBig);
  EXPECT_EQ(config_ok.worker_config.workers[0], DeviceFlag::kBandCPU);
  EXPECT_EQ(config_ok.worker_config.cpu_masks[0], CPUMaskFlag::kBandAll);
  EXPECT_EQ(config_ok.worker_config.num_threads[0], 1);
  EXPECT_EQ(config_ok.worker_config.allow_worksteal, true);
  EXPECT_EQ(config_ok.worker_config.availability_check_interval_ms, 100);
}

TEST(ConfigBuilderTest, DefaultValueTest) {
  RuntimeConfigBuilder b;
  RuntimeConfig config_ok = b.AddSchedulers({SchedulerType::kBandFixedWorker}).Build();
  EXPECT_EQ(config_ok.profile_config.online, true);
  EXPECT_EQ(config_ok.profile_config.num_warmups, 1);
  EXPECT_EQ(config_ok.profile_config.num_runs, 1);
  EXPECT_EQ(config_ok.profile_config.copy_computation_ratio[0], 30000);
  EXPECT_EQ(config_ok.profile_config.profile_data_path, "");
  EXPECT_EQ(config_ok.profile_config.smoothing_factor, 0.1f);
  EXPECT_EQ(config_ok.planner_config.log_path, "");
  EXPECT_EQ(config_ok.planner_config.schedulers[0], SchedulerType::kBandFixedWorker);
  EXPECT_EQ(config_ok.planner_config.schedule_window_size, INT_MAX);
  EXPECT_EQ(config_ok.planner_config.cpu_mask, CPUMaskFlag::kBandAll);
  EXPECT_EQ(config_ok.worker_config.workers[0], DeviceFlag::kBandCPU);
  EXPECT_EQ(config_ok.worker_config.workers[1], DeviceFlag::kBandGPU);
  EXPECT_EQ(config_ok.worker_config.workers[2], DeviceFlag::kBandDSP);
  EXPECT_EQ(config_ok.worker_config.workers[3], DeviceFlag::kBandNPU);
  EXPECT_EQ(config_ok.worker_config.cpu_masks[0], CPUMaskFlag::kBandAll);
  EXPECT_EQ(config_ok.worker_config.cpu_masks[1], CPUMaskFlag::kBandAll);
  EXPECT_EQ(config_ok.worker_config.cpu_masks[2], CPUMaskFlag::kBandAll);
  EXPECT_EQ(config_ok.worker_config.cpu_masks[3], CPUMaskFlag::kBandAll);
  EXPECT_EQ(config_ok.worker_config.num_threads[0], 1);
  EXPECT_EQ(config_ok.worker_config.num_threads[1], 1);
  EXPECT_EQ(config_ok.worker_config.num_threads[2], 1);
  EXPECT_EQ(config_ok.worker_config.num_threads[3], 1);
  EXPECT_EQ(config_ok.worker_config.allow_worksteal, false);
  EXPECT_EQ(config_ok.worker_config.availability_check_interval_ms, 30000);
  EXPECT_EQ(config_ok.subgraph_config.minimum_subgraph_size, 7);
  EXPECT_EQ(config_ok.subgraph_config.subgraph_preparation_type,
            SubgraphPreparationType::kBandMergeUnitSubgraph);
  EXPECT_EQ(config_ok.cpu_mask, CPUMaskFlag::kBandAll);
}

}  // namespace test
}  // namespace band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}