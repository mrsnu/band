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
    ErrorReporter* error_reporter /* = DefaultErrorReporter() */) {
  bool result = true;
  REPORT_IF_FALSE(ProfileConfigBuilder, num_warmups_ > 0);
  REPORT_IF_FALSE(ProfileConfigBuilder, num_runs_ > 0);
  return result;
}

bool LatencyProfileConfigBuilder::IsValid(
    ErrorReporter* error_reporter /* = DefaultErrorReporter()*/) {
  bool result = true;
  REPORT_IF_FALSE(LatencyProfileConfigBuilder,
                  smoothing_factor_ >= .0f && smoothing_factor_ <= 1.0f);
  return result;
}

bool ThermalProfileConfigBuilder::IsValid(
    ErrorReporter* error_reporter /* = DefaultErrorReporter()*/) {
  bool result = true;
  REPORT_IF_FALSE(ThermalProfileConfigBuilder, window_size_ > 0);
  return result;
}

bool DeviceConfigBuilder::IsValid(
    ErrorReporter* error_reporter /* = DefaultErrorReporter()*/) {
  bool result = true;
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
  REPORT_IF_FALSE(WorkerConfigBuilder, availability_check_interval_ms_ > 0);
  return result;
}

bool SubgraphConfigBuilder::IsValid(
  ErrorReporter* error_reporter /* = DefaultErrorReporter()*/) {
  bool result = true;
  REPORT_IF_FALSE(SubgraphConfigBuilder, minimum_subgraph_size_ > 0);
  REPORT_IF_FALSE(SubgraphConfigBuilder,
                  subgraph_preparation_type_ ==
                          SubgraphPreparationType::kNoFallbackSubgraph ||
                      subgraph_preparation_type_ ==
                          SubgraphPreparationType::kUnitSubgraph ||
                      subgraph_preparation_type_ ==
                          SubgraphPreparationType::kMergeUnitSubgraph);
  return result;
}

bool RuntimeConfigBuilder::IsValid(
    ErrorReporter* error_reporter /* = DefaultErrorReporter()*/) {
  bool result = true;
  REPORT_IF_FALSE(RuntimeConfigBuilder, cpu_mask_ == CPUMaskFlag::kAll ||
                                            cpu_mask_ == CPUMaskFlag::kLittle ||
                                            cpu_mask_ == CPUMaskFlag::kBig ||
                                            cpu_mask_ == CPUMaskFlag::kPrimary);

  // Independent validation
  REPORT_IF_FALSE(RuntimeConfigBuilder, profile_config_builder_.IsValid());
  REPORT_IF_FALSE(RuntimeConfigBuilder, planner_config_builder_.IsValid());
  REPORT_IF_FALSE(RuntimeConfigBuilder, worker_config_builder_.IsValid());
  REPORT_IF_FALSE(RuntimeConfigBuilder, subgraph_config_builder_.IsValid());

  return result;
}

ProfileConfig ProfileConfigBuilder::Build(
    ErrorReporter* error_reporter /* = DefaultErrorReporter()*/) {
  // This should not terminate the program. After employing abseil, Build()
  // should return error.
  if (!IsValid(error_reporter)) {
    abort();
  }
  ProfileConfig profile_config;
  profile_config.latency_config = latency_profile_builder_.Build();
  profile_config.thermal_config = thermal_profile_builder_.Build();
  profile_config.profile_path = profile_path_;
  profile_config.dump_path = dump_path_;
  profile_config.num_warmups = num_warmups_;
  profile_config.num_runs = num_runs_;
  return profile_config;
}

LatencyProfileConfig LatencyProfileConfigBuilder::Build(
    ErrorReporter* error_reporter /* = DefaultErrorReporter()*/) {
  // TODO(widiba03304): This should not terminate the program. After employing
  // abseil, Build() should return error.
  if (!IsValid(error_reporter)) {
    abort();
  }
  LatencyProfileConfig profile_config;
  profile_config.smoothing_factor = smoothing_factor_;
  return profile_config;
}

ThermalProfileConfig ThermalProfileConfigBuilder::Build(
    ErrorReporter* error_reporter /* = DefaultErrorReporter()*/) {
  // TODO(widiba03304): This should not terminate the program. After employing
  // abseil, Build() should return error.
  if (!IsValid(error_reporter)) {
    abort();
  }
  ThermalProfileConfig profile_config;
  profile_config.window_size = window_size_;
  return profile_config;
}

DeviceConfig DeviceConfigBuilder::Build(
    ErrorReporter* error_reporter /* = DefaultErrorReporter()*/) {
  // TODO(widiba03304): This should not terminate the program. After employing
  // abseil, Build() should return error.
  if (!IsValid(error_reporter)) {
    abort();
  }
  DeviceConfig device_config;
  device_config.cpu_therm_index = cpu_therm_index_;
  device_config.gpu_therm_index = gpu_therm_index_;
  device_config.dsp_therm_index = dsp_therm_index_;
  device_config.npu_therm_index = npu_therm_index_;
  device_config.target_therm_index = target_therm_index_;
  device_config.cpu_freq_path = cpu_freq_path_;
  device_config.gpu_freq_path = gpu_freq_path_;
  device_config.dsp_freq_path = dsp_freq_path_;
  device_config.npu_freq_path = npu_freq_path_;
  device_config.runtime_freq_path = runtime_freq_path_;
  device_config.latency_log_path = latency_log_path_;
  device_config.therm_log_path = therm_log_path_;
  device_config.freq_log_path = freq_log_path_;
  return device_config;
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
  planner_config.eta = eta_;
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
  worker_config.availability_check_interval_ms =
      availability_check_interval_ms_;
  return worker_config;
}

SubgraphConfig SubgraphConfigBuilder::Build(
    ErrorReporter* error_reporter /* = DefaultErrorReporter()*/) {
  // TODO(widiba03304): This should not terminate the program. After employing
  // abseil, Build() should return error.
  if (!IsValid(error_reporter)) {
    abort();
  }
  SubgraphConfig subgraph_config;
  subgraph_config.minimum_subgraph_size = minimum_subgraph_size_;
  subgraph_config.subgraph_preparation_type = subgraph_preparation_type_;
  return subgraph_config;
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
  DeviceConfig device_config = device_config_builder_.Build();
  PlannerConfig planner_config = planner_config_builder_.Build();
  WorkerConfig worker_config = worker_config_builder_.Build();
  SubgraphConfig subgraph_config = subgraph_config_builder_.Build();

  runtime_config.cpu_mask = cpu_mask_;
  runtime_config.profile_config = profile_config;
  runtime_config.device_config = device_config;
  runtime_config.planner_config = planner_config;
  runtime_config.worker_config = worker_config;
  runtime_config.subgraph_config = subgraph_config;
  return runtime_config;
}

}  // namespace band