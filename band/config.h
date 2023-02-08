#ifndef BAND_CONFIG_H_
#define BAND_CONFIG_H_

#include <string>
#include <vector>

#include "band/c/common.h"
#include "band/common.h"
#include "band/cpu.h"
#include "band/error_reporter.h"

namespace Band {

struct ProfileConfig {
  ProfileConfig() {
    copy_computation_ratio = std::vector<int>(kBandNumDevices, 0);
  }
  bool online = true;
  int num_warmups = 1;
  int num_runs = 1;
  std::vector<int> copy_computation_ratio;
  std::string profile_data_path = "";
  float smoothing_factor = 0.1;
};

struct PlannerConfig {
  int schedule_window_size = INT_MAX;
  std::vector<SchedulerType> schedulers;
  CPUMaskFlags cpu_mask = CPUMaskFlags::All;
  std::string log_path = "";
};

struct WorkerConfig {
  WorkerConfig() {
    // Add one default worker per device
    for (int i = 0; i < kBandNumDevices; i++) {
      workers.push_back(static_cast<BandDeviceFlags>(i));
    }
    cpu_masks = std::vector<CPUMaskFlags>(kBandNumDevices, CPUMaskFlags::All);
    num_threads = std::vector<int>(kBandNumDevices, 1);
  }
  std::vector<BandDeviceFlags> workers;
  std::vector<CPUMaskFlags> cpu_masks;
  std::vector<int> num_threads;
  bool allow_worksteal = false;
  int availability_check_interval_ms = 30000;
};

struct SubgraphConfig {
  int minimum_subgraph_size = 7;
  SubgraphPreparationType subgraph_preparation_type =
      SubgraphPreparationType::MergeUnitSubgraph;
};

struct RuntimeConfig {
  CPUMaskFlags cpu_mask;
  SubgraphConfig subgraph_config;
  ProfileConfig profile_config;
  PlannerConfig planner_config;
  WorkerConfig worker_config;

 private:
  friend class RuntimeConfigBuilder;
  RuntimeConfig() { cpu_mask = CPUMaskFlags::All; };
};

}  // namespace Band
#endif  // BAND_CONFIG_H_
