#include "band/config_builder.h"

#include <gtest/gtest.h>

namespace Band {
namespace Test {

TEST(ConfigBuilderTest, ProfileConfigBuilderTest) {
  ProfileConfigBuilder b;
  ProfileConfig config_ok =
      b.AddOnline(false).AddNumRuns(3).AddNumWarmups(3).Build();

  EXPECT_EQ(config_ok.online, false);
  EXPECT_EQ(config_ok.num_runs, 3);
  EXPECT_EQ(config_ok.num_warmups, 3);

  b.AddNumRuns(-1);
  EXPECT_FALSE(b.IsValid());
}

TEST(ConfigBuilderTest, PlannerConfigBuilderTest) {
  PlannerConfigBuilder b;
  PlannerConfig config_ok = b.AddLogPath("band/testdata/config.json")
                                .AddScheduleWindowSize(5)
                                .AddSchedulers({kBandFixedDevice})
                                .Build();
  EXPECT_EQ(config_ok.log_path, "band/testdata/config.json");
  EXPECT_EQ(config_ok.schedule_window_size, 5);
  EXPECT_EQ(config_ok.cpu_mask, kBandAll);

  b.AddScheduleWindowSize(-1);
  EXPECT_FALSE(b.IsValid());
}

TEST(ConfigBuilderTest, WorkerConfigBuilderTest) {
  WorkerConfigBuilder b;
  WorkerConfig config_ok = b.AddAllowWorkSteal(false)
                               .AddAvailabilityCheckIntervalMs(1000)
                               .AddAdditionalWorkers({kBandCPU, kBandDSP})
                               .AddCPUMasks({kBandAll, kBandAll})
                               .AddNumThreads({1, 1})
                               .Build();
  EXPECT_EQ(config_ok.allow_worksteal, false);
  EXPECT_EQ(config_ok.availability_check_interval_ms, 1000);
  EXPECT_EQ(config_ok.workers.size(), 6);
  EXPECT_EQ(config_ok.cpu_masks.size(), config_ok.workers.size());
  EXPECT_EQ(config_ok.num_threads.size(), config_ok.workers.size());

  b.AddAdditionalWorkers({kBandCPU});
  EXPECT_FALSE(b.IsValid());
  b.AddAdditionalWorkers({kBandCPU, kBandGPU});
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
          .AddProfileDataPath("band/testdata/config.json")
          .AddMinimumSubgraphSize(5)
          .AddSubgraphPreparationType(kBandMergeUnitSubgraph)
          .AddPlannerLogPath("band/testdata/config.json")
          .AddScheduleWindowSize(1)
          .AddSchedulers({kBandFixedDevice})
          .AddPlannerCPUMask(kBandBig)
          .AddAdditionalWorkers({})
          .AddWorkerCPUMasks({})
          .AddWorkerNumThreads({})
          .AddAllowWorkSteal(true)
          .AddAvailabilityCheckIntervalMs(100)
          .AddCPUMask(kBandPrimary)
          .Build();
  EXPECT_EQ(config_ok.profile_config.online, true);
  EXPECT_EQ(config_ok.profile_config.num_warmups, 1);
  EXPECT_EQ(config_ok.profile_config.num_runs, 1);
  EXPECT_EQ(
      config_ok.profile_config.copy_computation_ratio[0], 1);
  EXPECT_EQ(
      config_ok.profile_config.copy_computation_ratio[1], 2);
  EXPECT_EQ(
      config_ok.profile_config.copy_computation_ratio[2], 3);
  EXPECT_EQ(
      config_ok.profile_config.copy_computation_ratio[3], 4);
  EXPECT_EQ(config_ok.profile_config.smoothing_factor, 0.1f);
  EXPECT_EQ(config_ok.profile_config.profile_data_path, "band/testdata/config.json");
  EXPECT_EQ(config_ok.minimum_subgraph_size, 5);
  EXPECT_EQ(config_ok.subgraph_preparation_type, kBandMergeUnitSubgraph);
  EXPECT_EQ(config_ok.cpu_mask, kBandPrimary);
  EXPECT_EQ(config_ok.planner_config.log_path, "band/testdata/config.json");
  EXPECT_EQ(config_ok.planner_config.schedule_window_size, 1);
  EXPECT_EQ(config_ok.planner_config.schedulers[0], kBandFixedDevice);
  EXPECT_EQ(config_ok.planner_config.cpu_mask, kBandBig);
  EXPECT_EQ(config_ok.worker_config.workers[0], kBandCPU);
  EXPECT_EQ(config_ok.worker_config.cpu_masks[0], kBandAll);
  EXPECT_EQ(config_ok.worker_config.num_threads[0], 1);
  EXPECT_EQ(config_ok.worker_config.allow_worksteal, true);
  EXPECT_EQ(config_ok.worker_config.availability_check_interval_ms, 100);
}

TEST(ConfigBuilderTest, DefaultValueTest) {
  RuntimeConfigBuilder b;
  RuntimeConfig config_ok = b.AddSchedulers({kBandFixedDevice}).Build();
  EXPECT_EQ(config_ok.profile_config.online, true);
  EXPECT_EQ(config_ok.profile_config.num_warmups, 1);
  EXPECT_EQ(config_ok.profile_config.num_runs, 1);
  EXPECT_EQ(
      config_ok.profile_config.copy_computation_ratio[0],
      30000);
  EXPECT_EQ(config_ok.profile_config.profile_data_path, "");
  EXPECT_EQ(config_ok.profile_config.smoothing_factor, 0.1f);
  EXPECT_EQ(config_ok.planner_config.log_path, "");
  EXPECT_EQ(config_ok.planner_config.schedulers[0], kBandFixedDevice);
  EXPECT_EQ(config_ok.planner_config.schedule_window_size, INT_MAX);
  EXPECT_EQ(config_ok.planner_config.cpu_mask, kBandAll);
  EXPECT_EQ(config_ok.worker_config.workers[0], kBandCPU);
  EXPECT_EQ(config_ok.worker_config.workers[1], kBandGPU);
  EXPECT_EQ(config_ok.worker_config.workers[2], kBandDSP);
  EXPECT_EQ(config_ok.worker_config.workers[3], kBandNPU);
  EXPECT_EQ(config_ok.worker_config.cpu_masks[0], kBandAll);
  EXPECT_EQ(config_ok.worker_config.cpu_masks[1], kBandAll);
  EXPECT_EQ(config_ok.worker_config.cpu_masks[2], kBandAll);
  EXPECT_EQ(config_ok.worker_config.cpu_masks[3], kBandAll);
  EXPECT_EQ(config_ok.worker_config.num_threads[0], 1);
  EXPECT_EQ(config_ok.worker_config.num_threads[1], 1);
  EXPECT_EQ(config_ok.worker_config.num_threads[2], 1);
  EXPECT_EQ(config_ok.worker_config.num_threads[3], 1);
  EXPECT_EQ(config_ok.worker_config.allow_worksteal, false);
  EXPECT_EQ(config_ok.worker_config.availability_check_interval_ms, 30000);
  EXPECT_EQ(config_ok.minimum_subgraph_size, 7);
  EXPECT_EQ(config_ok.subgraph_preparation_type, kBandMergeUnitSubgraph);
  EXPECT_EQ(config_ok.cpu_mask, kBandAll);
}

}  // namespace Test
}  // namespace Band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}