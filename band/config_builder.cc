#include "band/config_builder.h"

#include "band/common.h"

namespace band {

#define REPORT_IF_FALSE(engine, expr)                            \
  do {                                                           \
    if (!(expr)) {                                               \
      return absl::InvalidArgumentError("[" #engine "] " #expr); \
    }                                                            \
  } while (0);

absl::Status ProfileConfigBuilder::IsValid() {
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
  return absl::OkStatus();
}

absl::Status PlannerConfigBuilder::IsValid() {
  REPORT_IF_FALSE(PlannerConfigBuilder, /*log_path_*/ true);  // Always true
  REPORT_IF_FALSE(PlannerConfigBuilder, schedule_window_size_ > 0);
  REPORT_IF_FALSE(PlannerConfigBuilder, schedulers_.size() > 0);
  REPORT_IF_FALSE(PlannerConfigBuilder, cpu_mask_ == CPUMaskFlag::kAll ||
                                            cpu_mask_ == CPUMaskFlag::kLittle ||
                                            cpu_mask_ == CPUMaskFlag::kBig ||
                                            cpu_mask_ == CPUMaskFlag::kPrimary);
  return absl::OkStatus();
}

absl::Status WorkerConfigBuilder::IsValid() {
  for (int i = 0; i < workers_.size(); i++) {
    REPORT_IF_FALSE(WorkerConfigBuilder,
                    workers_[i] == DeviceFlag::kCPU ||
                        workers_[i] == DeviceFlag::kGPU ||
                        workers_[i] == DeviceFlag::kDSP ||
                        workers_[i] == DeviceFlag::kNPU ||
                        workers_[i] == DeviceFlag::kNETWORK);
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
  return absl::OkStatus();
}

#ifdef BAND_TFLITE
absl::Status TfLiteBackendConfigBuilder::IsValid(
    ErrorReporter* error_reporter /* = DefaultErrorReporter()*/) {
  return absl::OkStatus();
}
#endif  // BAND_TFLITE

#ifdef BAND_GRPC
absl::Status GrpcBackendConfigBuilder::IsValid(
    ErrorReporter* error_reporter /* = DefaultErrorReporter()*/) {
  REPORT_IF_FALSE(GrpcBackendConfigBuilder, port_ > 0 && port_ < 65536);
  return absl::OkStatus();
}
#endif  // BAND_GRPC

absl::Status RuntimeConfigBuilder::IsValid() {
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
  RETURN_IF_ERROR(profile_config_builder_.IsValid());
  RETURN_IF_ERROR(planner_config_builder_.IsValid());
  RETURN_IF_ERROR(worker_config_builder_.IsValid());

  return absl::OkStatus();
}

absl::StatusOr<ProfileConfig> ProfileConfigBuilder::Build() {
  RETURN_IF_ERROR(IsValid());

  ProfileConfig profile_config;
  profile_config.online = online_;
  profile_config.num_warmups = num_warmups_;
  profile_config.num_runs = num_runs_;
  profile_config.copy_computation_ratio = copy_computation_ratio_;
  profile_config.smoothing_factor = smoothing_factor_;
  profile_config.profile_data_path = profile_data_path_;
  return profile_config;
}

absl::StatusOr<PlannerConfig> PlannerConfigBuilder::Build() {
  RETURN_IF_ERROR(IsValid());

  PlannerConfig planner_config;
  planner_config.log_path = log_path_;
  planner_config.schedule_window_size = schedule_window_size_;
  planner_config.schedulers = schedulers_;
  planner_config.cpu_mask = cpu_mask_;
  return planner_config;
}

absl::StatusOr<WorkerConfig> WorkerConfigBuilder::Build() {
  RETURN_IF_ERROR(IsValid());

  WorkerConfig worker_config;
  worker_config.workers = workers_;
  worker_config.cpu_masks = cpu_masks_;
  worker_config.num_threads = num_threads_;
  worker_config.allow_worksteal = allow_worksteal_;
  worker_config.availability_check_interval_ms =
      availability_check_interval_ms_;
  return worker_config;
}

#ifdef BAND_TFLITE
absl::StatusOr<std::shared_ptr<BackendConfig>>
TfLiteBackendConfigBuilder::Build(
    ErrorReporter* error_reporter /* = DefaultErrorReporter()*/) {
  // This should not terminate the program. After employing abseil, Build()
  // should return error.
  RETURN_IF_ERROR(IsValid());
  return std::make_shared<TfLiteBackendConfig>();
}
#endif  // BAND_TFLITE

#ifdef BAND_GRPC
absl::StatusOr<std::shared_ptr<BackendConfig>> GrpcBackendConfigBuilder::Build(
    ErrorReporter* error_reporter /* = DefaultErrorReporter()*/) {
  // This should not terminate the program. After employing abseil, Build()
  // should return error.
  RETURN_IF_ERROR(IsValid());
  return std::make_shared<GrpcBackendConfig>(host_, port_);
}
#endif  // BAND_GRPC

absl::StatusOr<ResourceMonitorConfig> ResourceMonitorConfigBuilder::Build() {
  RETURN_IF_ERROR(IsValid());

  ResourceMonitorConfig resource_monitor_config;
  resource_monitor_config.log_path = log_path_;
  resource_monitor_config.device_freq_paths = device_freq_paths_;
  resource_monitor_config.monitor_interval_ms = monitor_interval_ms_;
  return resource_monitor_config;
}

absl::Status ResourceMonitorConfigBuilder::IsValid() {
  REPORT_IF_FALSE(
      ResourceMonitorConfigBuilder,
      log_path_ == "" || log_path_.find(".json") != std::string::npos);
  return absl::OkStatus();
}

absl::StatusOr<RuntimeConfig> RuntimeConfigBuilder::Build() {
  RETURN_IF_ERROR(IsValid());
  RuntimeConfig runtime_config;
  runtime_config.subgraph_config = {minimum_subgraph_size_,
                                    subgraph_preparation_type_};
  runtime_config.cpu_mask = cpu_mask_;

  // No need to check the return value of Build() because it has been checked
  runtime_config.profile_config = profile_config_builder_.Build().value();
  runtime_config.planner_config = planner_config_builder_.Build().value();
  runtime_config.worker_config = worker_config_builder_.Build().value();
  runtime_config.resource_monitor_config =
      resource_monitor_config_builder_.Build().value();
#ifdef BAND_TFLITE
  runtime_config.backend_configs[BackendType::kTfLite] =
      tflite_backend_config_builder_.Build().value();
#endif  // BAND_TFLITE
#ifdef BAND_GRPC
  runtime_config.backend_configs[BackendType::kGrpc] =
      grpc_backend_config_builder_.Build().value();
#endif  // BAND_GRPC
  return runtime_config;
}
}  // namespace band