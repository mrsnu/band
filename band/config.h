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
};

struct InterpreterConfig {
  ProfileConfig profile_config;
  float smoothing_factor = 0.1;
  std::string log_path = "";
};

struct PlannerConfig {
  int schedule_window_size = INT_MAX;
  std::vector<BandSchedulerType> schedulers;
  BandCPUMaskFlags cpu_mask = kBandAll;
  std::string log_path = "";
};

struct WorkerConfig {
  WorkerConfig() {
    // Add one default worker per device
    for (int i = 0; i < kBandNumDevices; i++) {
      workers.push_back(static_cast<BandDeviceFlags>(i));
    }
    cpu_masks =
        std::vector<BandCPUMaskFlags>(kBandNumDevices, kBandAll);
    num_threads = std::vector<int>(kBandNumDevices, 1);
  }
  std::vector<BandDeviceFlags> workers;
  std::vector<BandCPUMaskFlags> cpu_masks;
  std::vector<int> num_threads;
  bool allow_worksteal = false;
  int availability_check_interval_ms = 30000;
};

struct RuntimeConfig {
  int minimum_subgraph_size;
  BandSubgraphPreparationType subgraph_preparation_type;
  InterpreterConfig interpreter_config;
  PlannerConfig planner_config;
  WorkerConfig worker_config;
 private:
  friend class RuntimeConfigBuilder;
  RuntimeConfig() {
    minimum_subgraph_size = 7;
    subgraph_preparation_type = kBandMergeUnitSubgraph;
  };
};

}  // namespace Band
#endif  // BAND_CONFIG_H_
