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
  friend class RuntimeConfigBuilder; // TODO: Find a safer way for RuntimeConfigBuilder to access variables
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
  ProfileConfigBuilder& AddProfileDataPath(std::string profile_data_path) {
    profile_data_path_ = profile_data_path;
    return *this;
  }
  ProfileConfigBuilder& AddSmoothingFactor(float smoothing_factor) {
    smoothing_factor_ = smoothing_factor;
    return *this;
  }

  ProfileConfig Build(ErrorReporter* error_reporter = DefaultErrorReporter());
  bool IsValid(ErrorReporter* error_reporter = DefaultErrorReporter());

 private:
  bool online_ = true;
  int num_warmups_ = 1;
  int num_runs_ = 1;
  std::vector<int> copy_computation_ratio_;
  std::string profile_data_path_ = "";
  float smoothing_factor_ = 0.1;
};

// Builder for creating PlannerConfig
class PlannerConfigBuilder {
  friend class RuntimeConfigBuilder;
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
  friend class RuntimeConfigBuilder;
 public:
  WorkerConfigBuilder() {
    for (int i = 0; i < kBandNumDevices; i++) {
      workers_.push_back(static_cast<BandDeviceFlags>(i));
    }
    cpu_masks_ = std::vector<BandCPUMaskFlags>(kBandNumDevices, kBandAll);
    num_threads_ = std::vector<int>(kBandNumDevices, 1);
  }
  WorkerConfigBuilder& AddWorkers(std::vector<BandDeviceFlags> workers) {
    workers_ = workers;
    return *this;
  }
  WorkerConfigBuilder& AddCPUMasks(std::vector<BandCPUMaskFlags> cpu_masks) {
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
  std::vector<BandDeviceFlags> workers_;
  std::vector<BandCPUMaskFlags> cpu_masks_;
  std::vector<int> num_threads_;
  bool allow_worksteal_ = false;
  int availability_check_interval_ms_ = 30000;
};

// Builder for creating ModelConfig. By default, no models are specified.
class ModelConfigBuilder {
  friend class RuntimeConfigBuilder;
  public:
    ModelConfigBuilder& AddModels(std::vector<std::string> models){
      models_ = models;
      return *this;
    }
    ModelConfigBuilder& AddPeriodsMs(std::vector<int> models_period_ms){
      models_period_ms_ = models_period_ms;
      return *this;
    }
    ModelConfigBuilder& AddBatchSizes(std::vector<int> models_batch_size){
      models_batch_size_ = models_batch_size;
      return *this;
    }
    ModelConfigBuilder& AddAssignedWorkers(std::vector<DeviceWorkerAffinityPair> models_assigned_worker){
      models_assigned_worker_ = models_assigned_worker;
      return *this;
    }
    ModelConfigBuilder& AddWorkersAffinity(std::vector<WorkerId> models_worker_affinity){
      models_worker_affinity_ = models_worker_affinity;
      return *this;
    }
    ModelConfigBuilder& AddSlosUs(std::vector<int64_t> models_slo_us){
      models_slo_us_ = models_slo_us;
      return *this;
    }
    ModelConfigBuilder& AddSlosScale(std::vector<float> models_slo_scale){
      models_slo_scale_ = models_slo_scale;
      return *this;
    }
    ModelConfig Build(ErrorReporter* error_reporter = DefaultErrorReporter());
    bool IsValid(ErrorReporter* error_reporter = DefaultErrorReporter());

  private:
    std::vector<std::string> models_;
    std::vector<int> models_period_ms_;
    std::vector<int> models_batch_size_;
    std::vector<DeviceWorkerAffinityPair> models_assigned_worker_;
    std::vector<int> models_worker_affinity_;
    std::vector<int64_t> models_slo_us_;
    std::vector<float> models_slo_scale_;
};

// Delegate for InterpreterConfigBuilder, PlannerConfigBuilder,
// and WorkerConfigBuilder
class RuntimeConfigBuilder {
 public:
  // Add ProfileConfig
  RuntimeConfigBuilder& AddOnline(bool online) {
    profile_config_builder_.AddOnline(online);
    return *this;
  }
  RuntimeConfigBuilder& AddNumWarmups(int num_warmups) {
    profile_config_builder_.AddNumWarmups(num_warmups);
    return *this;
  }
  RuntimeConfigBuilder& AddNumRuns(int num_runs) {
    profile_config_builder_.AddNumRuns(num_runs);
    return *this;
  }
  RuntimeConfigBuilder& AddCopyComputationRatio(
      std::vector<int> copy_computation_ratio) {
    profile_config_builder_.AddCopyComputationRatio(copy_computation_ratio);
    return *this;
  }

  // Add InterpreterConfig
  RuntimeConfigBuilder& AddSmoothingFactor(float smoothing_factor) {
    profile_config_builder_.AddSmoothingFactor(smoothing_factor);
    return *this;
  }
  RuntimeConfigBuilder& AddProfileDataPath(std::string profile_log_path) {
    profile_config_builder_.AddProfileDataPath(profile_log_path);
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
  RuntimeConfigBuilder& AddWorkers(std::vector<BandDeviceFlags> workers) {
    worker_config_builder_.AddWorkers(workers);
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
  RuntimeConfigBuilder& AddCPUMask(BandCPUMaskFlags cpu_mask) {
    cpu_mask_ = cpu_mask;
    return *this;
  }

  // Add ModelConfig
  RuntimeConfigBuilder& AddModels(std::vector<std::string> models){
    model_config_builder_.AddModels(models);
    return *this;
  }
  RuntimeConfigBuilder& AddPeriodsMs(std::vector<int> models_period_ms){
    model_config_builder_.AddPeriodsMs(models_period_ms);
    return *this;
  }
  RuntimeConfigBuilder& AddBatchSizes(std::vector<int> models_batch_size){
    model_config_builder_.AddBatchSizes(models_batch_size);
    return *this;
  }
  RuntimeConfigBuilder& AddAssignedWorkers(std::vector<DeviceWorkerAffinityPair> models_assigned_worker){
    model_config_builder_.AddAssignedWorkers(models_assigned_worker);
    return *this;
  }
  RuntimeConfigBuilder& AddSlosUs(std::vector<int64_t> models_slo_us){
    model_config_builder_.AddSlosUs(models_slo_us);
    return *this;
  }
  RuntimeConfigBuilder& AddSlosScale(std::vector<float> models_slo_scale){
    model_config_builder_.AddSlosScale(models_slo_scale);
    return *this;
  }

  RuntimeConfig Build(ErrorReporter* error_reporter = DefaultErrorReporter());
  bool IsValid(ErrorReporter* error_reporter = DefaultErrorReporter());

 private:
  ProfileConfigBuilder profile_config_builder_;
  PlannerConfigBuilder planner_config_builder_;
  WorkerConfigBuilder worker_config_builder_;
  ModelConfigBuilder model_config_builder_;
  int minimum_subgraph_size_ = 7;
  BandSubgraphPreparationType subgraph_preparation_type_ =
      kBandMergeUnitSubgraph;
  BandCPUMaskFlags cpu_mask_ = kBandAll;
};

}  // namespace Band

#endif  // BAND_CONFIG_BUILDER_H_
