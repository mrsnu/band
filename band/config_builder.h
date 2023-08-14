#ifndef BAND_CONFIG_BUILDER_H_
#define BAND_CONFIG_BUILDER_H_

#include <string>
#include <vector>

#include "band/common.h"
#include "band/config.h"
#include "band/device/cpu.h"
#include "band/error_reporter.h"

namespace band {

class LatencyProfileConfigBuilder {
  friend class RuntimeConfigBuilder;  // TODO: Find a safer way for
                                      // RuntimeConfigBuilder to access
                                      // variables
 public:
  LatencyProfileConfigBuilder() {
    copy_computation_ratio_ = std::vector<int>(EnumLength<DeviceFlag>(), 30000);
  }
  LatencyProfileConfigBuilder& AddOnline(bool online) {
    online_ = online;
    return *this;
  }
  LatencyProfileConfigBuilder& AddNumWarmups(int num_warmups) {
    num_warmups_ = num_warmups;
    return *this;
  }
  LatencyProfileConfigBuilder& AddNumRuns(int num_runs) {
    num_runs_ = num_runs;
    return *this;
  }
  LatencyProfileConfigBuilder& AddCopyComputationRatio(
      std::vector<int> copy_computation_ratio) {
    copy_computation_ratio_ = copy_computation_ratio;
    return *this;
  }
  LatencyProfileConfigBuilder& AddProfilePath(
      std::string profile_path) {
    profile_path_ = profile_path;
    return *this;
  }
  LatencyProfileConfigBuilder& AddSmoothingFactor(float smoothing_factor) {
    smoothing_factor_ = smoothing_factor;
    return *this;
  }

  LatencyProfileConfig Build(
      ErrorReporter* error_reporter = DefaultErrorReporter());
  bool IsValid(ErrorReporter* error_reporter = DefaultErrorReporter());

 private:
  bool online_ = true;
  int num_warmups_ = 1;
  int num_runs_ = 1;
  std::vector<int> copy_computation_ratio_;
  std::string profile_path_ = "";
  float smoothing_factor_ = 0.1;
};

class ThermalProfileConfigBuilder {
  friend class RuntimeConfigBuilder;  // TODO: Find a safer way for
                                      // RuntimeConfigBuilder to access
                                      // variables
 public:
  ThermalProfileConfigBuilder() {}
  ThermalProfileConfigBuilder& AddThermalProfileConfig(
      std::string profile_path) {
    profile_path_ = profile_path;
    return *this;
  }

  ThermalProfileConfig Build(
      ErrorReporter* error_reporter = DefaultErrorReporter());
  bool IsValid(ErrorReporter* error_reporter = DefaultErrorReporter());

 private:
  std::string profile_path_ = "";
};

// Builder for creating PlannerConfig
class PlannerConfigBuilder {
  friend class RuntimeConfigBuilder;

 public:
  PlannerConfigBuilder& AddScheduleWindowSize(int schedule_window_size) {
    schedule_window_size_ = schedule_window_size;
    return *this;
  }
  PlannerConfigBuilder& AddSchedulers(std::vector<SchedulerType> schedulers) {
    schedulers_ = schedulers;
    return *this;
  }
  PlannerConfigBuilder& AddCPUMask(CPUMaskFlag cpu_mask) {
    cpu_mask_ = cpu_mask;
    return *this;
  }
  PlannerConfigBuilder& AddLogPath(std::string log_path) {
    log_path_ = log_path;
    return *this;
  }

  PlannerConfig Build(ErrorReporter* error_reporter = DefaultErrorReporter());
  bool IsValid(ErrorReporter* error_reporter = DefaultErrorReporter());

 private:
  int schedule_window_size_ = INT_MAX;
  std::vector<SchedulerType> schedulers_;
  CPUMaskFlag cpu_mask_ = CPUMaskFlag::kAll;
  std::string log_path_ = "";
};

// Builder for creating WorkerConfig.
class WorkerConfigBuilder {
  friend class RuntimeConfigBuilder;

 public:
  WorkerConfigBuilder() {
    for (size_t i = 0; i < EnumLength<DeviceFlag>(); i++) {
      workers_.push_back(static_cast<DeviceFlag>(i));
    }
    cpu_masks_ =
        std::vector<CPUMaskFlag>(EnumLength<DeviceFlag>(), CPUMaskFlag::kAll);
    num_threads_ = std::vector<int>(EnumLength<DeviceFlag>(), 1);
  }
  WorkerConfigBuilder& AddWorkers(std::vector<DeviceFlag> workers) {
    workers_ = workers;
    return *this;
  }
  WorkerConfigBuilder& AddCPUMasks(std::vector<CPUMaskFlag> cpu_masks) {
    cpu_masks_ = cpu_masks;
    return *this;
  }
  WorkerConfigBuilder& AddNumThreads(std::vector<int> num_threads) {
    num_threads_ = num_threads;
    return *this;
  }
  WorkerConfigBuilder& AddAllowWorkSteal(bool allow_worksteal) {
    allow_worksteal_ = allow_worksteal;
    return *this;
  }
  WorkerConfigBuilder& AddAvailabilityCheckIntervalMs(
      int availability_check_interval_ms) {
    availability_check_interval_ms_ = availability_check_interval_ms;
    return *this;
  }
  WorkerConfig Build(ErrorReporter* error_reporter = DefaultErrorReporter());
  bool IsValid(ErrorReporter* error_reporter = DefaultErrorReporter());

 private:
  std::vector<DeviceFlag> workers_;
  std::vector<CPUMaskFlag> cpu_masks_;
  std::vector<int> num_threads_;
  bool allow_worksteal_ = false;
  int availability_check_interval_ms_ = 30000;
};

class ResourceMonitorConfigBuilder {
  friend class RuntimeConfigBuilder;

 public:
  ResourceMonitorConfigBuilder& AddResourceMonitorLogPath(
      std::string log_path) {
    resource_monitor_log_path_ = log_path;
    return *this;
  }
  ResourceMonitorConfigBuilder& AddResourceMonitorDeviceFreqPath(
      DeviceFlag device, std::string device_freq_path) {
    device_freq_paths_.insert({device, device_freq_path});
    return *this;
  }
  ResourceMonitorConfigBuilder& AddResourceMonitorIntervalMs(
      int monitor_interval_ms) {
    monitor_interval_ms_ = monitor_interval_ms;
    return *this;
  }

  ResourceMonitorConfig Build(
      ErrorReporter* error_reporter = DefaultErrorReporter());
  bool IsValid(ErrorReporter* error_reporter = DefaultErrorReporter());

 private:
  std::string resource_monitor_log_path_ = "";
  std::map<DeviceFlag, std::string> device_freq_paths_;
  int monitor_interval_ms_ = 10;
};

// Delegate for ConfigBuilders
class RuntimeConfigBuilder {
 public:
  // Add LatencyProfileConfig
  RuntimeConfigBuilder& AddOnline(bool online) {
    latency_profile_config_builder_.AddOnline(online);
    return *this;
  }
  RuntimeConfigBuilder& AddNumWarmups(int num_warmups) {
    latency_profile_config_builder_.AddNumWarmups(num_warmups);
    return *this;
  }
  RuntimeConfigBuilder& AddNumRuns(int num_runs) {
    latency_profile_config_builder_.AddNumRuns(num_runs);
    return *this;
  }
  RuntimeConfigBuilder& AddCopyComputationRatio(
      std::vector<int> copy_computation_ratio) {
    latency_profile_config_builder_.AddCopyComputationRatio(copy_computation_ratio);
    return *this;
  }

  RuntimeConfigBuilder& AddSmoothingFactor(float smoothing_factor) {
    latency_profile_config_builder_.AddSmoothingFactor(smoothing_factor);
    return *this;
  }
  RuntimeConfigBuilder& AddProfilePath(std::string profile_log_path) {
    latency_profile_config_builder_.AddProfilePath(profile_log_path);
    return *this;
  }

  // Add PlannerConfig
  RuntimeConfigBuilder& AddPlannerLogPath(std::string planner_log_path) {
    planner_config_builder_.AddLogPath(planner_log_path);
    return *this;
  }
  RuntimeConfigBuilder& AddScheduleWindowSize(int schedule_window_size) {
    planner_config_builder_.AddScheduleWindowSize(schedule_window_size);
    return *this;
  }
  RuntimeConfigBuilder& AddSchedulers(std::vector<SchedulerType> schedulers) {
    planner_config_builder_.AddSchedulers(schedulers);
    return *this;
  }
  RuntimeConfigBuilder& AddPlannerCPUMask(CPUMaskFlag cpu_masks) {
    planner_config_builder_.AddCPUMask(cpu_masks);
    return *this;
  }

  // Add WorkerConfig
  RuntimeConfigBuilder& AddWorkers(std::vector<DeviceFlag> workers) {
    worker_config_builder_.AddWorkers(workers);
    return *this;
  }
  RuntimeConfigBuilder& AddWorkerCPUMasks(std::vector<CPUMaskFlag> cpu_masks) {
    worker_config_builder_.AddCPUMasks(cpu_masks);
    return *this;
  }
  RuntimeConfigBuilder& AddWorkerNumThreads(std::vector<int> num_threads) {
    worker_config_builder_.AddNumThreads(num_threads);
    return *this;
  }
  RuntimeConfigBuilder& AddAllowWorkSteal(bool allow_worksteal) {
    worker_config_builder_.AddAllowWorkSteal(allow_worksteal);
    return *this;
  }
  RuntimeConfigBuilder& AddAvailabilityCheckIntervalMs(
      int32_t availability_check_interval_ms) {
    worker_config_builder_.AddAvailabilityCheckIntervalMs(
        availability_check_interval_ms);
    return *this;
  }
  RuntimeConfigBuilder& AddResourceMonitorLogPath(
      std::string resource_monitor_log_path) {
    device_config_builder_.AddResourceMonitorLogPath(resource_monitor_log_path);
    return *this;
  }
  RuntimeConfigBuilder& AddResourceMonitorDeviceFreqPath(
      DeviceFlag device, std::string device_freq_path) {
    device_config_builder_.AddResourceMonitorDeviceFreqPath(device,
                                                            device_freq_path);
    return *this;
  }
  RuntimeConfigBuilder& AddResourceMonitorIntervalMs(int monitor_interval_ms) {
    device_config_builder_.AddResourceMonitorIntervalMs(monitor_interval_ms);
    return *this;
  }
  RuntimeConfigBuilder& AddMinimumSubgraphSize(int minimum_subgraph_size) {
    minimum_subgraph_size_ = minimum_subgraph_size;
    return *this;
  }
  RuntimeConfigBuilder& AddSubgraphPreparationType(
      SubgraphPreparationType subgraph_preparation_type) {
    subgraph_preparation_type_ = subgraph_preparation_type;
    return *this;
  }
  RuntimeConfigBuilder& AddCPUMask(CPUMaskFlag cpu_mask) {
    cpu_mask_ = cpu_mask;
    return *this;
  }

  RuntimeConfig Build(ErrorReporter* error_reporter = DefaultErrorReporter());
  bool IsValid(ErrorReporter* error_reporter = DefaultErrorReporter());

 private:
  LatencyProfileConfigBuilder latency_profile_config_builder_;
  ThermalProfileConfigBuilder thermal_profile_config_builder_;
  PlannerConfigBuilder planner_config_builder_;
  WorkerConfigBuilder worker_config_builder_;
  ResourceMonitorConfigBuilder device_config_builder_;
  int minimum_subgraph_size_ = 7;
  SubgraphPreparationType subgraph_preparation_type_ =
      SubgraphPreparationType::kMergeUnitSubgraph;
  CPUMaskFlag cpu_mask_ = CPUMaskFlag::kAll;
};

}  // namespace band

#endif  // BAND_CONFIG_BUILDER_H_
