#include "band/config_builder.h"

#include <gtest/gtest.h>

namespace band {
namespace test {

TEST(ConfigBuilderTest, ProfileConfigBuilderTest) {
  ProfileConfigBuilder b;
  auto config = b.AddOnline(false)
                    .AddNumRuns(3)
                    .AddNumWarmups(3)
                    .AddProfileDataPath("hello")
                    .Build();
  EXPECT_EQ(config.status(), absl::OkStatus());
  ProfileConfig config_ok = config.value();
  EXPECT_EQ(config_ok.online, false);
  EXPECT_EQ(config_ok.num_runs, 3);
  EXPECT_EQ(config_ok.num_warmups, 3);

  b.AddNumRuns(-1);
  EXPECT_FALSE(b.Build().ok());
  b.AddNumRuns(1);
  b.AddOnline(true);
  EXPECT_TRUE(b.Build().ok());
}

TEST(ConfigBuilderTest, PlannerConfigBuilderTest) {
  PlannerConfigBuilder b;
  auto config = b.AddLogPath("band/test/data/config.json")
                    .AddScheduleWindowSize(5)
                    .AddSchedulers({SchedulerType::kFixedWorker})
                    .Build();
  EXPECT_EQ(config.status(), absl::OkStatus());
  PlannerConfig config_ok = config.value();
  EXPECT_EQ(config_ok.log_path, "band/test/data/config.json");
  EXPECT_EQ(config_ok.schedule_window_size, 5);
  EXPECT_EQ(config_ok.cpu_mask, CPUMaskFlag::kAll);

  b.AddScheduleWindowSize(-1);
  EXPECT_FALSE(b.Build().ok());
}

TEST(ConfigBuilderTest, WorkerConfigBuilderTest) {
  WorkerConfigBuilder b;
  auto config = b.AddAllowWorkSteal(false)
                    .AddAvailabilityCheckIntervalMs(1000)
                    .AddWorkers({DeviceFlag::kCPU, DeviceFlag::kDSP})
                    .AddCPUMasks({CPUMaskFlag::kAll, CPUMaskFlag::kAll})
                    .AddNumThreads({1, 1})
                    .Build();
  EXPECT_EQ(config.status(), absl::OkStatus());
  WorkerConfig config_ok = config.value();
  EXPECT_EQ(config_ok.allow_worksteal, false);
  EXPECT_EQ(config_ok.availability_check_interval_ms, 1000);
  EXPECT_EQ(config_ok.workers.size(), 2);
  EXPECT_EQ(config_ok.cpu_masks.size(), config_ok.workers.size());
  EXPECT_EQ(config_ok.num_threads.size(), config_ok.workers.size());

  b.AddWorkers({DeviceFlag::kCPU});

  EXPECT_FALSE(b.Build().ok());
  b.AddWorkers({DeviceFlag::kCPU, DeviceFlag::kGPU});
  EXPECT_TRUE(b.Build().ok());
}

TEST(ConfigBuilderTest, RuntimeConfigBuilderTest) {
  RuntimeConfigBuilder b;
  auto config = b.AddOnline(true)
                    .AddNumWarmups(1)
                    .AddNumRuns(1)
                    .AddSmoothingFactor(0.1)
                    .AddProfileDataPath("band/test/data/config.json")
                    .AddMinimumSubgraphSize(5)
                    .AddSubgraphPreparationType(
                        SubgraphPreparationType::kMergeUnitSubgraph)
                    .AddPlannerLogPath("band/test/data/config.json")
                    .AddScheduleWindowSize(1)
                    .AddSchedulers({SchedulerType::kFixedWorker})
                    .AddPlannerCPUMask(CPUMaskFlag::kBig)
                    .AddWorkers({})
                    .AddWorkerCPUMasks({})
                    .AddWorkerNumThreads({})
                    .AddAllowWorkSteal(true)
                    .AddAvailabilityCheckIntervalMs(100)
                    .AddCPUMask(CPUMaskFlag::kPrimary)
                    .Build();
  EXPECT_EQ(config.status(), absl::OkStatus());
  RuntimeConfig config_ok = config.value();
  EXPECT_EQ(config_ok.profile_config.online, true);
  EXPECT_EQ(config_ok.profile_config.num_warmups, 1);
  EXPECT_EQ(config_ok.profile_config.num_runs, 1);
  EXPECT_EQ(config_ok.profile_config.smoothing_factor, 0.1f);
  EXPECT_EQ(config_ok.profile_config.profile_data_path,
            "band/test/data/config.json");
  EXPECT_EQ(config_ok.subgraph_config.minimum_subgraph_size, 5);
  EXPECT_EQ(config_ok.subgraph_config.subgraph_preparation_type,
            SubgraphPreparationType::kMergeUnitSubgraph);
  EXPECT_EQ(config_ok.cpu_mask, CPUMaskFlag::kPrimary);
  EXPECT_EQ(config_ok.planner_config.log_path, "band/test/data/config.json");
  EXPECT_EQ(config_ok.planner_config.schedule_window_size, 1);
  EXPECT_EQ(config_ok.planner_config.schedulers[0],
            SchedulerType::kFixedWorker);
  EXPECT_EQ(config_ok.planner_config.cpu_mask, CPUMaskFlag::kBig);
  EXPECT_EQ(config_ok.worker_config.workers[0], DeviceFlag::kCPU);
  EXPECT_EQ(config_ok.worker_config.cpu_masks[0], CPUMaskFlag::kAll);
  EXPECT_EQ(config_ok.worker_config.num_threads[0], 1);
  EXPECT_EQ(config_ok.worker_config.allow_worksteal, true);
  EXPECT_EQ(config_ok.worker_config.availability_check_interval_ms, 100);
}

TEST(ConfigBuilderTest, DefaultValueTest) {
  RuntimeConfigBuilder b;
  auto config = b.AddSchedulers({SchedulerType::kFixedWorker}).Build();
  EXPECT_EQ(config.status(), absl::OkStatus());
  RuntimeConfig config_ok = config.value();
  EXPECT_EQ(config_ok.profile_config.online, true);
  EXPECT_EQ(config_ok.profile_config.num_warmups, 1);
  EXPECT_EQ(config_ok.profile_config.num_runs, 1);
  EXPECT_EQ(config_ok.profile_config.profile_data_path, "");
  EXPECT_EQ(config_ok.profile_config.smoothing_factor, 0.1f);
  EXPECT_EQ(config_ok.planner_config.log_path, "");
  EXPECT_EQ(config_ok.planner_config.schedulers[0],
            SchedulerType::kFixedWorker);
  EXPECT_EQ(config_ok.planner_config.schedule_window_size, std::numeric_limits<int>::max());
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
  EXPECT_EQ(config_ok.worker_config.allow_worksteal, false);
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