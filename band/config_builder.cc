#include "band/config_builder.h"

#include "config_builder.h"

namespace band {

#define REPORT_IF_FALSE(engine, expr)                            \
  do {                                                           \
    result &= (expr);                                            \
    if (!(expr)) {                                               \
      BAND_REPORT_ERROR(error_reporter, "[" #engine "] " #expr); \
    }                                                            \
  } while (0);

bool ProfileConfigBuilder::IsValid(
    ErrorReporter* error_reporter /* = DefaultErrorReporter()*/) {
  bool result = true;
  REPORT_IF_FALSE(ProfileConfigBuilder,
                  online_ == true || online_ == false);  // Always true
  REPORT_IF_FALSE(ProfileConfigBuilder, num_warmups_ > 0);
  REPORT_IF_FALSE(ProfileConfigBuilder, num_runs_ > 0);
  REPORT_IF_FALSE(ProfileConfigBuilder,
                  copy_computation_ratio_.size() == EnumLength<DeviceFlag>());
  for (int i = 0; i < EnumLength<DeviceFlag>(); i++) {
    REPORT_IF_FALSE(ProfileConfigBuilder, copy_computation_ratio_[i] >= 0);
  }
  REPORT_IF_FALSE(ProfileConfigBuilder,
                  smoothing_factor_ >= .0f && smoothing_factor_ <= 1.0f);
  if (online_ == false) {
    REPORT_IF_FALSE(ProfileConfigBuilder, profile_data_path_ != "");
  }
  return result;
}

bool PlannerConfigBuilder::IsValid(
    ErrorReporter* error_reporter /* = DefaultErrorReporter()*/) {
  bool result = true;
  REPORT_IF_FALSE(PlannerConfigBuilder, /*log_path_*/ true);  // Always true
  REPORT_IF_FALSE(PlannerConfigBuilder, schedule_window_size_ > 0);
  REPORT_IF_FALSE(PlannerConfigBuilder, schedulers_.size() > 0);
  REPORT_IF_FALSE(PlannerConfigBuilder, cpu_mask_ == CPUMaskFlag::kAll ||
                                            cpu_mask_ == CPUMaskFlag::kLittle ||
                                            cpu_mask_ == CPUMaskFlag::kBig ||
                                            cpu_mask_ == CPUMaskFlag::kPrimary);
  return result;
}

bool WorkerConfigBuilder::IsValid(
    ErrorReporter* error_reporter /* = DefaultErrorReporter()*/) {
  bool result = true;
  for (int i = 0; i < workers_.size(); i++) {
    REPORT_IF_FALSE(WorkerConfigBuilder, workers_[i] == DeviceFlag::kCPU ||
                                             workers_[i] == DeviceFlag::kGPU ||
                                             workers_[i] == DeviceFlag::kDSP ||
                                             workers_[i] == DeviceFlag::kNPU);
  }
  REPORT_IF_FALSE(WorkerConfigBuilder, cpu_masks_.size() == workers_.size());
  for (int i = 0; i < workers_.size(); i++) {
    REPORT_IF_FALSE(WorkerConfigBuilder,
                    cpu_masks_[i] == CPUMaskFlag::kAll ||
                        cpu_masks_[i] == CPUMaskFlag::kLittle ||
                        cpu_masks_[i] == CPUMaskFlag::kBig ||
                        cpu_masks_[i] == CPUMaskFlag::kPrimary);
  }
  REPORT_IF_FALSE(WorkerConfigBuilder, num_threads_.size() == workers_.size());
  for (int i = 0; i < workers_.size(); i++) {
    REPORT_IF_FALSE(WorkerConfigBuilder, num_threads_[i] >= /*or >?*/ 0);
  }
  REPORT_IF_FALSE(WorkerConfigBuilder,
                  allow_worksteal_ == true || allow_worksteal_ == false);
  REPORT_IF_FALSE(WorkerConfigBuilder, availability_check_interval_ms_ > 0);
  return result;
}

bool RuntimeConfigBuilder::IsValid(
    ErrorReporter* error_reporter /* = DefaultErrorReporter()*/) {
  bool result = true;
  REPORT_IF_FALSE(RuntimeConfigBuilder, minimum_subgraph_size_ > 0);
  REPORT_IF_FALSE(RuntimeConfigBuilder,
                  subgraph_preparation_type_ ==
                          SubgraphPreparationType::kNoFallbackSubgraph ||
                      subgraph_preparation_type_ ==
                          SubgraphPreparationType::kFallbackPerWorker ||
                      subgraph_preparation_type_ ==
                          SubgraphPreparationType::kUnitSubgraph ||
                      subgraph_preparation_type_ ==
                          SubgraphPreparationType::kMergeUnitSubgraph);
  REPORT_IF_FALSE(RuntimeConfigBuilder, cpu_mask_ == CPUMaskFlag::kAll ||
                                            cpu_mask_ == CPUMaskFlag::kLittle ||
                                            cpu_mask_ == CPUMaskFlag::kBig ||
                                            cpu_mask_ == CPUMaskFlag::kPrimary);

  // Independent validation
  REPORT_IF_FALSE(RuntimeConfigBuilder, profile_config_builder_.IsValid());
  REPORT_IF_FALSE(RuntimeConfigBuilder, planner_config_builder_.IsValid());
  REPORT_IF_FALSE(RuntimeConfigBuilder, worker_config_builder_.IsValid());

  return result;
}

ProfileConfig ProfileConfigBuilder::Build(
    ErrorReporter* error_reporter /* = DefaultErrorReporter()*/) {
  // TODO(widiba03304): This should not terminate the program. After employing
  // abseil, Build() should return error.
  if (!IsValid(error_reporter)) {
    abort();
  }
  ProfileConfig profile_config;
  profile_config.online = online_;
  profile_config.num_warmups = num_warmups_;
  profile_config.num_runs = num_runs_;
  profile_config.copy_computation_ratio = copy_computation_ratio_;
  profile_config.smoothing_factor = smoothing_factor_;
  profile_config.profile_data_path = profile_data_path_;
  return profile_config;
}

PlannerConfig PlannerConfigBuilder::Build(
    ErrorReporter* error_reporter /* = DefaultErrorReporter()*/) {
  // TODO(widiba03304): This should not terminate the program. After employing
  // abseil, Build() should return error.
  if (!IsValid(error_reporter)) {
    abort();
  }
  PlannerConfig planner_config;
  planner_config.log_path = log_path_;
  planner_config.schedule_window_size = schedule_window_size_;
  planner_config.schedulers = schedulers_;
  planner_config.cpu_mask = cpu_mask_;
  return planner_config;
}

WorkerConfig WorkerConfigBuilder::Build(
    ErrorReporter* error_reporter /* = DefaultErrorReporter()*/) {
  // TODO(widiba03304): This should not terminate the program. After employing
  // abseil, Build() should return error.
  if (!IsValid(error_reporter)) {
    abort();
  }
  WorkerConfig worker_config;
  worker_config.workers = workers_;
  worker_config.cpu_masks = cpu_masks_;
  worker_config.num_threads = num_threads_;
  worker_config.allow_worksteal = allow_worksteal_;
  worker_config.availability_check_interval_ms =
      availability_check_interval_ms_;
  return worker_config;
}

ResourceMonitorConfig ResourceMonitorConfigBuilder::Build(
    ErrorReporter* error_reporter) {
  if (!IsValid(error_reporter)) {
    abort();
  }

  ResourceMonitorConfig device_config;
  device_config.resource_monitor_log_path = resource_monitor_log_path_;
  device_config.device_freq_paths = device_freq_paths_;
  device_config.monitor_interval_ms = monitor_interval_ms_;
  return device_config;
}

bool ResourceMonitorConfigBuilder::IsValid(ErrorReporter* error_reporter) {
  bool result = true;

  REPORT_IF_FALSE(
      ResourceMonitorConfigBuilder,
      resource_monitor_log_path_ == "" ||
          resource_monitor_log_path_.find(".json") != std::string::npos);
  return result;
}

RuntimeConfig RuntimeConfigBuilder::Build(
    ErrorReporter* error_reporter /* = DefaultErrorReporter()*/) {
  // TODO(widiba03304): This should not terminate the program. After employing
  // abseil, Build() should return error.
  if (!IsValid(error_reporter)) {
    abort();
  }

  RuntimeConfig runtime_config;
  ProfileConfig profile_config = profile_config_builder_.Build();
  PlannerConfig planner_config = planner_config_builder_.Build();
  WorkerConfig worker_config = worker_config_builder_.Build();
  ResourceMonitorConfig device_config = device_config_builder_.Build();
  runtime_config.subgraph_config = {minimum_subgraph_size_,
                                    subgraph_preparation_type_};

  runtime_config.cpu_mask = cpu_mask_;
  runtime_config.profile_config = profile_config;
  runtime_config.planner_config = planner_config;
  runtime_config.worker_config = worker_config;
  runtime_config.device_config = device_config;
  return runtime_config;
}

}  // namespace band