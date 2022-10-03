#ifndef BAND_CONFIG_BUILDER_H_
#define BAND_CONFIG_BUILDER_H_

#include <string>
#include <vector>

#include "band/common.h"
#include "band/config.h"
#include "band/cpu.h"
#include "band/error_reporter.h"

namespace Band {

class ProfileConfigBuilder {
 public:
  ProfileConfigBuilder() {
    copy_computation_ratio_ = std::vector<int>(kBandNumDevices, 30000);
  }
  ProfileConfigBuilder& AddOnline(bool online) {
    online_ = online;
    return *this;
  }
  ProfileConfigBuilder& AddNumWarmups(int num_warmups) {
    num_warmups_ = num_warmups;
    return *this;
  }
  ProfileConfigBuilder& AddNumRuns(int num_runs) {
    num_runs_ = num_runs;
    return *this;
  }
  ProfileConfigBuilder& AddCopyComputationRatio(
      std::vector<int> copy_computation_ratio) {
    copy_computation_ratio_ = copy_computation_ratio;
    return *this;
  }

  ProfileConfig Build(ErrorReporter* error_reporter = DefaultErrorReporter());
  bool IsValid(ErrorReporter* error_reporter = DefaultErrorReporter());

 private:
  bool online_ = true;
  int num_warmups_ = 1;
  int num_runs_ = 1;
  std::vector<int> copy_computation_ratio_;
};

// Builder for creating InterpreterConfig, and delegates ProfileConfigBuilder.
class InterpreterConfigBuilder {
 public:
  InterpreterConfigBuilder() {}
  // Delegation on ProfileConfigBuilder.
  InterpreterConfigBuilder& AddOnline(bool online) {
    profile_config_builder_.AddOnline(online);
    return *this;
  }
  InterpreterConfigBuilder& AddNumWarmups(int num_warmups) {
    profile_config_builder_.AddNumWarmups(num_warmups);
    return *this;
  }
  InterpreterConfigBuilder& AddNumRuns(int num_runs) {
    profile_config_builder_.AddNumRuns(num_runs);
    return *this;
  }
  InterpreterConfigBuilder& AddCopyComputationRatio(
      std::vector<int> copy_computation_ratio) {
    profile_config_builder_.AddCopyComputationRatio(copy_computation_ratio);
    return *this;
  }
  InterpreterConfigBuilder& AddSmoothingFactor(float smoothing_factor) {
    smoothing_factor_ = smoothing_factor;
    return *this;
  }
  InterpreterConfigBuilder& AddLogPath(std::string log_path) {
    log_path_ = log_path;
    return *this;
  }

  InterpreterConfig Build(ErrorReporter* error_reporter = DefaultErrorReporter());
  bool IsValid(ErrorReporter* error_reporter = DefaultErrorReporter());

 private:
  // Profile config
  ProfileConfigBuilder profile_config_builder_;
  float smoothing_factor_ = 0.1;
  std::string log_path_;
  // Interpreter config
  int minimum_subgraph_size_ = 7;
  BandSubgraphPreparationType subgraph_preparation_type_ =
      kBandMergeUnitSubgraph;
};

// Builder for creating PlannerConfig
class PlannerConfigBuilder {
 public:
  PlannerConfigBuilder& AddScheduleWindowSize(int schedule_window_size) {
    schedule_window_size_ = schedule_window_size;
    return *this;
  }
  PlannerConfigBuilder& AddSchedulers(
      std::vector<BandSchedulerType> schedulers) {
    schedulers_ = schedulers;
    return *this;
  }
  PlannerConfigBuilder& AddCPUMask(BandCPUMaskFlags cpu_mask) {
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
  std::vector<BandSchedulerType> schedulers_;
  BandCPUMaskFlags cpu_mask_ = kBandAll;
  std::string log_path_ = "";
};

// Builder for creating WorkerConfig.
class WorkerConfigBuilder {
 public:
  WorkerConfigBuilder() {
    for (int i = 0; i < kBandNumDevices; i++) {
      workers_.push_back(static_cast<BandDeviceFlags>(i));
    }
    cpu_masks_ = std::vector<BandCPUMaskFlags>(kBandNumDevices, kBandAll);
    num_threads_ = std::vector<int>(kBandNumDevices, 1 /*or 0?*/);
  }
  // To keep idempotency, `AddAdditionalWorkers` create a default workers
  // and add the given workers to the default.
  WorkerConfigBuilder& AddAdditionalWorkers(
      std::vector<BandDeviceFlags> additional_workers) {
    std::vector<BandDeviceFlags> temp_workers;
    // Default workers
    for (int i = 0; i < kBandNumDevices; i++) {
      temp_workers.push_back(static_cast<BandDeviceFlags>(i));
    }
    temp_workers.insert(temp_workers.end(), additional_workers.begin(),
                        additional_workers.end());
    workers_ = temp_workers;
    return *this;
  }
  // To keep idempotency, `AddCPUMasks` create a default cpu_masks
  // and add the given cpu_masks to the default.
  WorkerConfigBuilder& AddCPUMasks(std::vector<BandCPUMaskFlags> cpu_masks) {
    std::vector<BandCPUMaskFlags> temp_cpu_masks;
    temp_cpu_masks = std::vector<BandCPUMaskFlags>(kBandNumDevices, kBandAll);
    temp_cpu_masks.insert(temp_cpu_masks.end(), cpu_masks.begin(),
                          cpu_masks.end());
    cpu_masks_ = temp_cpu_masks;
    return *this;
  }
  WorkerConfigBuilder& AddNumThreads(std::vector<int> num_threads) {
    std::vector<int> temp_num_threads;
    // TODO(widiba03304): #205
    temp_num_threads = std::vector<int>(kBandNumDevices, 1 /*or 0?*/);
    temp_num_threads.insert(temp_num_threads.end(), num_threads.begin(),
                            num_threads.end());
    num_threads_ = temp_num_threads;
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
  std::vector<BandDeviceFlags> workers_;
  std::vector<BandCPUMaskFlags> cpu_masks_;
  std::vector<int> num_threads_;
  bool allow_worksteal_ = false;
  int availability_check_interval_ms_ = 30000;
};

// Delegate for InterpreterConfigBuilder, PlannerConfigBuilder,
// and WorkerConfigBuilder
class RuntimeConfigBuilder {
 public:
  // Add ProfileConfig
  RuntimeConfigBuilder& AddOnline(bool online) {
    interpreter_config_builder_.AddOnline(online);
    return *this;
  }
  RuntimeConfigBuilder& AddNumWarmups(int num_warmups) {
    interpreter_config_builder_.AddNumWarmups(num_warmups);
    return *this;
  }
  RuntimeConfigBuilder& AddNumRuns(int num_runs) {
    interpreter_config_builder_.AddNumRuns(num_runs);
    return *this;
  }
  RuntimeConfigBuilder& AddCopyComputationRatio(
      std::vector<int> copy_computation_ratio) {
    interpreter_config_builder_.AddCopyComputationRatio(copy_computation_ratio);
    return *this;
  }

  // Add InterpreterConfig
  RuntimeConfigBuilder& AddSmoothingFactor(float smoothing_factor) {
    interpreter_config_builder_.AddSmoothingFactor(smoothing_factor);
    return *this;
  }
  RuntimeConfigBuilder& AddProfileLogPath(std::string profile_log_path) {
    interpreter_config_builder_.AddLogPath(profile_log_path);
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
  RuntimeConfigBuilder& AddSchedulers(
      std::vector<BandSchedulerType> schedulers) {
    planner_config_builder_.AddSchedulers(schedulers);
    return *this;
  }
  RuntimeConfigBuilder& AddPlannerCPUMask(BandCPUMaskFlags cpu_masks) {
    planner_config_builder_.AddCPUMask(cpu_masks);
    return *this;
  }

  // Add WorkerConfig
  RuntimeConfigBuilder& AddAdditionalWorkers(
      std::vector<BandDeviceFlags> workers) {
    worker_config_builder_.AddAdditionalWorkers(workers);
    return *this;
  }
  RuntimeConfigBuilder& AddWorkerCPUMasks(
      std::vector<BandCPUMaskFlags> cpu_masks) {
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
      BandSubgraphPreparationType subgraph_preparation_type) {
    subgraph_preparation_type_ = subgraph_preparation_type;
    return *this;
  }

  RuntimeConfig Build(ErrorReporter* error_reporter = DefaultErrorReporter());
  bool IsValid(ErrorReporter* error_reporter = DefaultErrorReporter());

 private:
  InterpreterConfigBuilder interpreter_config_builder_;
  PlannerConfigBuilder planner_config_builder_;
  WorkerConfigBuilder worker_config_builder_;
  int minimum_subgraph_size_ = 7;
  BandSubgraphPreparationType subgraph_preparation_type_ =
      kBandMergeUnitSubgraph;
};

}  // namespace Band

#endif  // BAND_CONFIG_BUILDER_H_
