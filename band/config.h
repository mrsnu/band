#ifndef BAND_CONFIG_H_
#define BAND_CONFIG_H_

#include <string>
#include <vector>

#include "band/common.h"
#include "band/error_reporter.h"

#include <limits>

namespace band {

struct ProfileConfig {
  ProfileConfig() {
    copy_computation_ratio = std::vector<int>(EnumLength<DeviceFlag>(), 0);
  }
  bool online = true;
  int num_warmups = 1;
  int num_runs = 1;
  std::vector<int> copy_computation_ratio;
  std::string profile_data_path = "";
  float smoothing_factor = 0.1;
};

struct PlannerConfig {
  int schedule_window_size = std::numeric_limits<int>::max();
  std::vector<SchedulerType> schedulers;
  CPUMaskFlag cpu_mask = CPUMaskFlag::kAll;
  std::string log_path = "";
};

struct WorkerConfig {
  WorkerConfig() {
    // Add one default worker per device
    for (size_t i = 0; i < EnumLength<DeviceFlag>(); i++) {
      workers.push_back(static_cast<DeviceFlag>(i));
    }
    cpu_masks =
        std::vector<CPUMaskFlag>(EnumLength<DeviceFlag>(), CPUMaskFlag::kAll);
    num_threads = std::vector<int>(EnumLength<DeviceFlag>(), 1);
  }
  std::vector<DeviceFlag> workers;
  std::vector<CPUMaskFlag> cpu_masks;
  std::vector<int> num_threads;
  bool allow_worksteal = false;
  int availability_check_interval_ms = 30000;
};

struct SubgraphConfig {
  int minimum_subgraph_size = 7;
  SubgraphPreparationType subgraph_preparation_type =
      SubgraphPreparationType::kMergeUnitSubgraph;
};

struct RuntimeConfig {
  CPUMaskFlag cpu_mask;
  SubgraphConfig subgraph_config;
  ProfileConfig profile_config;
  PlannerConfig planner_config;
  WorkerConfig worker_config;

 private:
  friend class RuntimeConfigBuilder;
  RuntimeConfig() { cpu_mask = CPUMaskFlag::kAll; };
};

}  // namespace band
#endif  // BAND_CONFIG_H_
