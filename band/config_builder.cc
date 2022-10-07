#include "band/config_builder.h"

#include "band/macros.h"

namespace Band {

#define REPORT_IF_FALSE(context, expr)                            \
  do {                                                            \
    result &= (expr);                                             \
    if (!(expr)) {                                                \
      BAND_REPORT_ERROR(error_reporter, "[" #context "] " #expr); \
    }                                                             \
  } while (0);

bool ProfileConfigBuilder::IsValid(
    ErrorReporter* error_reporter /* = DefaultErrorReporter()*/) {
  bool result = true;
  REPORT_IF_FALSE(ProfileConfigBuilder,
                  online_ == true || online_ == false);  // Always true
  REPORT_IF_FALSE(ProfileConfigBuilder, num_warmups_ > 0);
  REPORT_IF_FALSE(ProfileConfigBuilder, num_runs_ > 0);
  REPORT_IF_FALSE(ProfileConfigBuilder,
                  copy_computation_ratio_.size() == kBandNumDevices);
  for (int i = 0; i < kBandNumDevices; i++) {
    REPORT_IF_FALSE(ProfileConfigBuilder, copy_computation_ratio_[i] >= 0);
  }
  REPORT_IF_FALSE(ProfileConfigBuilder,
                  smoothing_factor_ >= .0f && smoothing_factor_ <= 1.0f);
  REPORT_IF_FALSE(ProfileConfigBuilder,
                  /*profile_data_path_*/ true);  // Always true
  return result;
}

bool PlannerConfigBuilder::IsValid(
    ErrorReporter* error_reporter /* = DefaultErrorReporter()*/) {
  bool result = true;
  REPORT_IF_FALSE(PlannerConfigBuilder, /*log_path_*/ true);  // Always true
  REPORT_IF_FALSE(PlannerConfigBuilder, schedule_window_size_ > 0);
  REPORT_IF_FALSE(PlannerConfigBuilder, schedulers_.size() > 0);
  REPORT_IF_FALSE(PlannerConfigBuilder,
                  cpu_mask_ == kBandAll || cpu_mask_ == kBandLittle ||
                      cpu_mask_ == kBandBig || cpu_mask_ == kBandPrimary);
  return result;
}

bool WorkerConfigBuilder::IsValid(
    ErrorReporter* error_reporter /* = DefaultErrorReporter()*/) {
  bool result = true;
  for (int i = 0; i < workers_.size(); i++) {
    REPORT_IF_FALSE(WorkerConfigBuilder,
                    workers_[i] == kBandCPU || workers_[i] == kBandGPU ||
                        workers_[i] == kBandDSP || workers_[i] == kBandNPU);
  }
  REPORT_IF_FALSE(WorkerConfigBuilder, cpu_masks_.size() == workers_.size());
  for (int i = 0; i < workers_.size(); i++) {
    REPORT_IF_FALSE(WorkerConfigBuilder, cpu_masks_[i] == kBandAll ||
                                             cpu_masks_[i] == kBandLittle ||
                                             cpu_masks_[i] == kBandBig ||
                                             cpu_masks_[i] == kBandPrimary);
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

bool ModelConfigBuilder::IsValid(
  ErrorReporter* error_reporter /* = DefaultErrorReporter()*/) {
    bool result = true;
    if(models_.size()>0){
      if(models_period_ms_.size() > 0) {
        REPORT_IF_FALSE(ModelConfigBuilder, models_.size() == models_period_ms_.size());
      } 
      if(models_assigned_worker_.size() > 0) {
        REPORT_IF_FALSE(ModelConfigBuilder, models_.size() == models_assigned_worker_.size());
        for(int i=0; i<models_.size(); i++){
          REPORT_IF_FALSE(ModelConfigBuilder, models_assigned_worker_[i].device == kBandCPU || 
                                          models_assigned_worker_[i].device == kBandGPU || 
                                          models_assigned_worker_[i].device == kBandNPU || 
                                          models_assigned_worker_[i].device == kBandDSP);
        }
      } 
      if(models_batch_size_.size() > 0) {
        REPORT_IF_FALSE(ModelConfigBuilder, models_.size() == models_batch_size_.size());
      } 
      if(models_slo_us_.size() > 0) {
        REPORT_IF_FALSE(ModelConfigBuilder, models_.size() == models_slo_us_.size());
      } 
      if(models_slo_scale_.size() > 0) {
        REPORT_IF_FALSE(ModelConfigBuilder, models_.size() == models_slo_scale_.size());
      } 
    }
    
    return result;
  }

bool RuntimeConfigBuilder::IsValid(
    ErrorReporter* error_reporter /* = DefaultErrorReporter()*/) {
  bool result = true;
  REPORT_IF_FALSE(RuntimeConfigBuilder, minimum_subgraph_size_ > 0);
  REPORT_IF_FALSE(RuntimeConfigBuilder,
                  subgraph_preparation_type_ == kBandNoFallbackSubgraph ||
                      subgraph_preparation_type_ == kBandFallbackPerDevice ||
                      subgraph_preparation_type_ == kBandUnitSubgraph ||
                      subgraph_preparation_type_ == kBandMergeUnitSubgraph);
  REPORT_IF_FALSE(RuntimeConfigBuilder,
                  cpu_mask_ == kBandAll || cpu_mask_ == kBandLittle ||
                      cpu_mask_ == kBandBig || cpu_mask_ == kBandPrimary);

  // Independent validation
  REPORT_IF_FALSE(RuntimeConfigBuilder, profile_config_builder_.IsValid());
  REPORT_IF_FALSE(RuntimeConfigBuilder, planner_config_builder_.IsValid());
  REPORT_IF_FALSE(RuntimeConfigBuilder, worker_config_builder_.IsValid());
  REPORT_IF_FALSE(RuntimeConfigBuilder, model_config_builder_.IsValid());

  // Interdependent validation
  // Check that worker affinity matches the number of workers defined
  std::map<BandDeviceFlags, int> NumWorkersPerDevice;
  for(int device=kBandCPU; device != kBandNumDevices; device++){
    NumWorkersPerDevice.emplace(static_cast<BandDeviceFlags>(device), 0);
  }
  for(int i=0; i<worker_config_builder_.workers_.size(); i++){
    NumWorkersPerDevice[worker_config_builder_.workers_[i]] ++;
  }
  for(int i=0; i<model_config_builder_.models_assigned_worker_.size(); i++){
    DeviceWorkerAffinityPair pair = model_config_builder_.models_assigned_worker_[i];
    if(pair.worker < NumWorkersPerDevice[pair.device]) {
      NumWorkersPerDevice[pair.device]--;
    } else{
      result = false;
      break;
    }
  }
  
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

ModelConfig ModelConfigBuilder::Build(
  ErrorReporter* error_reporter /* = DefaultErrorReporter()*/) {
  if (!IsValid(error_reporter)) {
    abort();
  }
  ModelConfig model_config;
  model_config.models = models_;
  model_config.models_assigned_worker = models_assigned_worker_;
  model_config.models_batch_size = models_batch_size_;
  model_config.models_period_ms = models_period_ms_;
  model_config.models_slo_scale = models_slo_scale_;
  model_config.models_slo_us = models_slo_us_;
  return model_config;
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
  ModelConfig model_config = model_config_builder_.Build();

  runtime_config.minimum_subgraph_size = minimum_subgraph_size_;
  runtime_config.subgraph_preparation_type = subgraph_preparation_type_;
  runtime_config.cpu_mask = cpu_mask_;
  runtime_config.profile_config = profile_config;
  runtime_config.planner_config = planner_config;
  runtime_config.worker_config = worker_config;
  runtime_config.model_configs = model_config;
  return runtime_config;
}

}  // namespace Band