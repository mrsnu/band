#ifndef BAND_CONFIG_H_
#define BAND_CONFIG_H_

#include <string>
#include <vector>

#include "band/c/common.h"
#include "band/common.h"
#include "band/cpu.h"
#include "band/error_reporter.h"

#include <json/json.h>

namespace Band {
struct ProfileConfig {
  ProfileConfig() {
    copy_computation_ratio = std::vector<int>(kBandNumDevices, 0);
  }
  bool online = true;
  int num_warmups = 1;
  int num_runs = 1;
  std::vector<int> copy_computation_ratio;

  // below moved from InterpreterConfig
  int default_copy_computation_ratio = 1000;
  float profile_smoothing_factor = 0.1;
  std::string profile_data_path;
};

struct InterpreterConfig {
  ProfileConfig profile_config;
  int minimum_subgraph_size = 7;
  std::string subgraph_preparation_type = "merge_unit_subgraph";
  BandCPUMaskFlags cpu_masks = kBandAll;
  int num_threads = -1;
};

struct PlannerConfig {
  std::string log_path;
  int schedule_window_size = INT_MAX;
  std::vector<BandSchedulerType> schedulers;
  BandCPUMaskFlags cpu_masks = kBandAll;
};

struct WorkerConfig {
  WorkerConfig() {
    // Add one default worker per device
    for (int i = 0; i < kBandNumDevices; i++) {
      workers.push_back(static_cast<BandDeviceFlags>(i));
    }
    cpu_masks =
        std::vector<BandCPUMaskFlags>(kBandNumDevices, kBandNumCpuMasks);
    num_threads = std::vector<int>(kBandNumDevices, 0);
  }
  std::vector<BandDeviceFlags> workers;
  std::vector<BandCPUMaskFlags> cpu_masks;
  std::vector<int> num_threads;
  bool allow_worksteal = false;
  int32_t availability_check_interval_ms = 30000;
};

struct RuntimeConfig {
  InterpreterConfig interpreter_config;
  PlannerConfig planner_config;
  WorkerConfig worker_config;
};

class ErrorReporter;
// Parse runtime config from a json file path
BandStatus ParseRuntimeConfigFromJsonObject(
    const Json::Value &root, RuntimeConfig &runtime_config,
    ErrorReporter *error_reporter = DefaultErrorReporter());

BandStatus ParseRuntimeConfigFromJson(
    std::string json_fname, RuntimeConfig &runtime_config,
    ErrorReporter *error_reporter = DefaultErrorReporter());

BandStatus ParseRuntimeConfigFromJson(
    const void *buffer, size_t buffer_length, RuntimeConfig &runtime_config,
    ErrorReporter *error_reporter = DefaultErrorReporter());

// Check if the keys exist in the config
BandStatus
ValidateJsonConfig(const Json::Value &json_config,
                   std::vector<std::string> keys,
                   ErrorReporter *error_reporter = DefaultErrorReporter());
} // namespace Band
#endif // BAND_CONFIG_H_
