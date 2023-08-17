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
  LatencyProfileConfigBuilder& AddSmoothingFactor(float smoothing_factor) {
    smoothing_factor_ = smoothing_factor;
    return *this;
  }

  LatencyProfileConfig Build(
      ErrorReporter* error_reporter = DefaultErrorReporter());
  bool IsValid(ErrorReporter* error_reporter = DefaultErrorReporter());

 private:
  float smoothing_factor_ = 0.1f;
};

class FrequencyLatencyProfileConfigBuilder {
  friend class RuntimeConfigBuilder;

 public:
  FrequencyLatencyProfileConfigBuilder& AddSmoothingFactor(
      float smoothing_factor) {
    smoothing_factor_ = smoothing_factor;
    return *this;
  }

  FrequencyLatencyProfileConfig Build(
      ErrorReporter* error_reporter = DefaultErrorReporter());
  bool IsValid(ErrorReporter* error_reporter = DefaultErrorReporter());

 private:
  float smoothing_factor_ = 0.1f;
};

class ThermalProfileConfigBuilder {
  friend class RuntimeConfigBuilder;

 public:
  ThermalProfileConfigBuilder& AddCPUIndex(size_t cpu_index) {
    cpu_index_ = cpu_index;
    return *this;
  }

  ThermalProfileConfigBuilder& AddGPUIndex(size_t gpu_index) {
    gpu_index_ = gpu_index;
    return *this;
  }

  ThermalProfileConfigBuilder& AddDSPIndex(size_t dsp_index) {
    dsp_index_ = dsp_index;
    return *this;
  }

  ThermalProfileConfigBuilder& AddNPUIndex(size_t npu_index) {
    npu_index_ = npu_index;
    return *this;
  }

  ThermalProfileConfig Build(
      ErrorReporter* error_reporter = DefaultErrorReporter());
  bool IsValid(ErrorReporter* error_reporter = DefaultErrorReporter());

 private:
  size_t cpu_index_ = -1;
  size_t gpu_index_ = -1;
  size_t dsp_index_ = -1;
  size_t npu_index_ = -1;
};

class ProfileConfigBuilder {
  friend class RuntimeConfigBuilder;

 public:
  ProfileConfigBuilder& AddLatencySmoothingFactor(float smoothing_factor) {
    latency_profile_builder_.AddSmoothingFactor(smoothing_factor);
    return *this;
  }
  ProfileConfigBuilder& AddFrequencySmoothingFactor(float smoothing_factor) {
    freq_latency_profile_builder_.AddSmoothingFactor(smoothing_factor);
    return *this;
  }
  ProfileConfigBuilder& AddNumWarmups(size_t num_warmups) {
    num_warmups_ = num_warmups;
    return *this;
  }
  ProfileConfigBuilder& AddNumRuns(size_t num_runs) {
    num_runs_ = num_runs;
    return *this;
  }
  ProfileConfigBuilder& AddProfilePath(std::string profile_path) {
    profile_path_ = profile_path;
    return *this;
  }

  ProfileConfig Build(ErrorReporter* error_reporter = DefaultErrorReporter());
  bool IsValid(ErrorReporter* error_reporter = DefaultErrorReporter());

 private:
  LatencyProfileConfigBuilder latency_profile_builder_;
  FrequencyLatencyProfileConfigBuilder freq_latency_profile_builder_;
  ThermalProfileConfigBuilder thermal_profile_builder_;
  std::string profile_path_ = "";
  size_t num_warmups_ = 1;
  size_t num_runs_ = 1;
};

class DeviceConfigBuilder {
  friend class RuntimeConfigBuilder;

 public:
  DeviceConfigBuilder& AddCPUThermalIndex(size_t cpu_therm_index) {
    cpu_therm_index_ = cpu_therm_index;
    return *this;
  }

  DeviceConfigBuilder& AddGPUThermalIndex(size_t gpu_therm_index) {
    gpu_therm_index_ = gpu_therm_index;
    return *this;
  }

  DeviceConfigBuilder& AddDSPThermalIndex(size_t dsp_therm_index) {
    dsp_therm_index_ = dsp_therm_index;
    return *this;
  }

  DeviceConfigBuilder& AddNPUThermalIndex(size_t npu_therm_index) {
    npu_therm_index_ = npu_therm_index;
    return *this;
  }

  DeviceConfigBuilder& AddCPUFreqPath(std::string cpu_freq_path) {
    cpu_freq_path_ = cpu_freq_path;
    return *this;
  }

  DeviceConfigBuilder& AddGPUFreqPath(std::string gpu_freq_path) {
    gpu_freq_path_ = gpu_freq_path;
    return *this;
  }

  DeviceConfigBuilder& AddDSPFreqPath(std::string dsp_freq_path) {
    dsp_freq_path_ = dsp_freq_path;
    return *this;
  }

  DeviceConfigBuilder& AddNPUFreqPath(std::string npu_freq_path) {
    npu_freq_path_ = npu_freq_path;
    return *this;
  }

  DeviceConfigBuilder& AddLatencyDumpPath(std::string latency_log_path) {
    latency_log_path_ = latency_log_path;
    return *this;
  }

  DeviceConfigBuilder& AddThermalDumpPath(std::string therm_log_path) {
    therm_log_path_ = therm_log_path;
    return *this;
  }

  DeviceConfigBuilder& AddFrequencyDumpPath(std::string freq_log_path) {
    freq_log_path_ = freq_log_path;
    return *this;
  }

  DeviceConfig Build(ErrorReporter* error_reporter = DefaultErrorReporter());
  bool IsValid(ErrorReporter* error_reporter = DefaultErrorReporter());

 private:
  size_t cpu_therm_index_ = -1;
  size_t gpu_therm_index_ = -1;
  size_t dsp_therm_index_ = -1;
  size_t npu_therm_index_ = -1;
  std::string cpu_freq_path_ = "";
  std::string gpu_freq_path_ = "";
  std::string dsp_freq_path_ = "";
  std::string npu_freq_path_ = "";
  std::string latency_log_path_ = "";
  std::string therm_log_path_ = "";
  std::string freq_log_path_ = "";
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

// Delegate for ConfigBuilders
class RuntimeConfigBuilder {
 public:
  // Add ProfileConfig
  RuntimeConfigBuilder& AddLatencySmoothingFactor(float smoothing_factor) {
    profile_config_builder_.AddLatencySmoothingFactor(smoothing_factor);
    return *this;
  }
  RuntimeConfigBuilder& AddFrequencyLatencySmoothingFactor(
      float smoothing_factor) {
    profile_config_builder_.AddFrequencySmoothingFactor(smoothing_factor);
    return *this;
  }
  RuntimeConfigBuilder& AddNumWarmups(size_t num_warmups) {
    profile_config_builder_.AddNumWarmups(num_warmups);
    return *this;
  }
  RuntimeConfigBuilder& AddNumRuns(size_t num_runs) {
    profile_config_builder_.AddNumRuns(num_runs);
    return *this;
  }
  RuntimeConfigBuilder& AddProfilePath(std::string profile_path) {
    profile_config_builder_.AddProfilePath(profile_path);
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
  ProfileConfigBuilder profile_config_builder_;
  DeviceConfigBuilder device_config_builder_;
  PlannerConfigBuilder planner_config_builder_;
  WorkerConfigBuilder worker_config_builder_;
  int minimum_subgraph_size_ = 7;
  SubgraphPreparationType subgraph_preparation_type_ =
      SubgraphPreparationType::kMergeUnitSubgraph;
  CPUMaskFlag cpu_mask_ = CPUMaskFlag::kAll;
};

}  // namespace band

#endif  // BAND_CONFIG_BUILDER_H_
