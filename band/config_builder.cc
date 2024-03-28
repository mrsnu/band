// Copyright 2023 Seoul National University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
  return absl::OkStatus();
}

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
  return runtime_config;
}

RuntimeConfig RuntimeConfigBuilder::GetDefaultConfig() {
  RuntimeConfigBuilder builder;

  builder.AddPlannerLogPath("/data/local/tmp/log.json");
  builder.AddSchedulers({SchedulerType::kHeterogeneousEarliestFinishTime});
  builder.AddMinimumSubgraphSize(7);
  builder.AddSubgraphPreparationType(
      SubgraphPreparationType::kMergeUnitSubgraph);
  builder.AddCPUMask(CPUMaskFlag::kAll);
  builder.AddPlannerCPUMask(CPUMaskFlag::kPrimary);
  builder.AddWorkers(
      {DeviceFlag::kCPU, DeviceFlag::kGPU, DeviceFlag::kDSP, DeviceFlag::kNPU});
  builder.AddWorkerNumThreads({1, 1, 1, 1});
  builder.AddWorkerCPUMasks({CPUMaskFlag::kBig, CPUMaskFlag::kBig,
                             CPUMaskFlag::kBig, CPUMaskFlag::kBig});
  builder.AddSmoothingFactor(0.1f);
  builder.AddProfileDataPath("/data/local/tmp/profile.json");
  builder.AddOnline(true);
  builder.AddNumWarmups(1);
  builder.AddNumRuns(1);
  builder.AddAllowWorkSteal(true);
  builder.AddAvailabilityCheckIntervalMs(30000);
  builder.AddScheduleWindowSize(10);
  return builder.Build().value();
}

}  // namespace band