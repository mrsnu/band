#ifndef BAND_CONFIG_H_
#define BAND_CONFIG_H_

#include <limits>
#include <string>
#include <vector>

#include "band/common.h"
#include "band/error_reporter.h"

namespace band {

struct FrequencyLatencyProfileConfig {
  float smoothing_factor = 0.1f;
};

struct LatencyProfileConfig {
  float smoothing_factor = 0.1f;
};

struct ThermalProfileConfig {
};

struct DeviceConfig {
  size_t cpu_therm_index = -1;
  size_t gpu_therm_index = -1;
  size_t dsp_therm_index = -1;
  size_t npu_therm_index = -1;

  std::string cpu_freq_path = "";
  std::string gpu_freq_path = "";
  std::string dsp_freq_path = "";
  std::string npu_freq_path = "";

  std::string latency_log_path = "";
  std::string therm_log_path = "";
  std::string freq_log_path = "";
};

struct ProfileConfig {
  LatencyProfileConfig latency_config;
  ThermalProfileConfig thermal_config;
  FrequencyLatencyProfileConfig frequency_latency_config;
  std::string profile_path = "";
  size_t num_warmups = 1;
  size_t num_runs = 1;
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
  DeviceConfig device_config;

 private:
  friend class RuntimeConfigBuilder;
  RuntimeConfig() { cpu_mask = CPUMaskFlag::kAll; };
};

}  // namespace band
#endif  // BAND_CONFIG_H_
