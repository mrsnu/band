#ifndef BAND_CONFIG_H_
#define BAND_CONFIG_H_

#include <limits>
#include <string>
#include <vector>
#include <memory>

#include "band/common.h"
#include "band/error_reporter.h"

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

struct ResourceMonitorConfig {
  std::string log_path = "";
  std::map<DeviceFlag, std::string> device_freq_paths;
  int monitor_interval_ms = 10;
};

struct BackendConfig {};

#ifdef BAND_TFLITE
struct TfLiteBackendConfig : public BackendConfig {};
#endif  // BAND_TFLITE

#ifdef BAND_GRPC
struct GrpcBackendConfig : public BackendConfig {
  GrpcBackendConfig(std::string host, int port) : host(host), port(port) {}
  std::string host = "localhost";
  int port = 50051;
};
#endif  // BAND_GRPC

struct RuntimeConfig {
  CPUMaskFlag cpu_mask;
  SubgraphConfig subgraph_config;
  ProfileConfig profile_config;
  PlannerConfig planner_config;
  WorkerConfig worker_config;
  std::map<BackendType, std::shared_ptr<BackendConfig>> backend_configs;
  ResourceMonitorConfig resource_monitor_config;

 private:
  friend class RuntimeConfigBuilder;
  RuntimeConfig() { cpu_mask = CPUMaskFlag::kAll; };
};

}  // namespace band
#endif  // BAND_CONFIG_H_
