#include "band/tool/engine_runner.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <thread>

#include "absl/strings/str_format.h"
#include "band/config_builder.h"
#include "band/logger.h"
#include "band/model.h"
#include "band/profiler.h"
#include "band/tensor.h"
#include "band/time.h"
#include "band/tool/benchmark.h"
#include "band/tool/graph_runner.h"

namespace band {
namespace tool {

EngineRunner::EngineRunner(BackendType target_backend)
    : target_backend_(target_backend) {}

EngineRunner::~EngineRunner() {
  if (engine_) {
    delete engine_;
  }

  if (runtime_config_) {
    delete runtime_config_;
  }

  for (auto& model : registered_models_) {
    delete model.second;
  }
}

absl::StatusOr<const Model*> EngineRunner::GetModel(
    const std::string& model_key) const {
  std::lock_guard<std::mutex> lock(model_mutex_);
  if (registered_models_.find(model_key) == registered_models_.end()) {
    return absl::NotFoundError("Model not found");
  } else {
    return registered_models_.at(model_key);
  }
}

absl::Status EngineRunner::LoadRunnerConfigs(const Json::Value& root) {
  if (!json::Validate(root, {"running_time_ms", "graph_workloads", "models"})) {
    return absl::InvalidArgumentError(
        "Please check if argument `running_time_ms` and `graph_workloads` are "
        "given");
  }

  json::AssignIfValid(running_time_ms_, root, "running_time_ms");

  if (running_time_ms_ <= 0) {
    return absl::InvalidArgumentError(
        "Please check if argument running_time_ms >= 0");
  }

  for (auto& model_key : root["models"].getMemberNames()) {
    if (!root["models"][model_key].isString()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Please check if model path for model %s is "
                          "given",
                          model_key.c_str()));
    }
    std::string model_path = root["models"][model_key].asString();
    if (registered_models_.find(model_key) == registered_models_.end()) {
      Model* model = new Model();
      RETURN_IF_ERROR(model->FromPath(target_backend_, model_path.c_str()));
      RETURN_IF_ERROR(engine_->RegisterModel(model));
      registered_models_[model_path] = model;
    }
  }

  for (auto& graph : root["graph_workloads"]) {
    std::unique_ptr<GraphRunner> graph_runner =
        std::make_unique<GraphRunner>(target_backend_, *this);
    RETURN_IF_ERROR(graph_runner->Initialize(graph));
    children_.push_back(graph_runner.release());
  }

  return absl::OkStatus();
}

absl::StatusOr<RuntimeConfig*> EngineRunner::LoadRuntimeConfigs(
    const Json::Value& root) {
  if (!json::Validate(root, {"schedulers"})) {
    return absl::InvalidArgumentError(
        "Please check if argument `schedulers` is given");
  }

  RuntimeConfigBuilder builder;
  // Profile config
  {
    if (root["profile_warmup_runs"].isNumeric()) {
      builder.AddNumWarmups(root["profile_warmup_runs"].asFloat());
    }
    if (root["profile_num_runs"].isInt()) {
      builder.AddNumRuns(root["profile_num_runs"].asInt());
    }
    if (root["profile_copy_computation_ratio"].isNumeric()) {
      std::vector<int> copy_computation_ratio;
      for (auto ratio : root["profile_copy_computation_ratio"]) {
        copy_computation_ratio.push_back(ratio.asInt());
      }
      builder.AddCopyComputationRatio(copy_computation_ratio);
    }
    if (root["profile_smoothing_factor"].isNumeric()) {
      builder.AddSmoothingFactor(root["profile_smoothing_factor"].asFloat());
    }
    if (root["profile_data_path"].isString()) {
      builder.AddProfileDataPath(root["profile_data_path"].asCString());
    }
  }

  // Planner config
  {
    if (root["schedule_window_size"].isInt()) {
      builder.AddScheduleWindowSize(root["schedule_window_size"].asInt());
    }

    std::vector<SchedulerType> schedulers;
    for (auto scheduler : root["schedulers"]) {
      if (!scheduler.isString()) {
        return absl::InvalidArgumentError(
            "Please check if given scheduler is valid");
      }
      schedulers.push_back(FromString<SchedulerType>(scheduler.asCString()));
    }
    builder.AddSchedulers(schedulers);

    if (root["cpu_masks"].isString()) {
      builder.AddCPUMask(
          FromString<CPUMaskFlag>(root["cpu_masks"].asCString()));
    }

    if (root["log_path"].isString()) {
      builder.AddPlannerLogPath(root["log_path"].asCString());
    }
  }

  // Worker config
  {
    if (!root["workers"].isNull()) {
      std::vector<DeviceFlag> workers;
      std::vector<CPUMaskFlag> cpu_masks;
      std::vector<int> num_threads;

      for (auto worker : root["workers"]) {
        if (worker["device"].isString()) {
          workers.push_back(
              FromString<DeviceFlag>(worker["device"].asCString()));
        }
        if (worker["num_threads"].isInt()) {
          num_threads.push_back(worker["num_threads"].asInt());
        }
        if (worker["cpu_masks"].isString()) {
          cpu_masks.push_back(
              FromString<CPUMaskFlag>(worker["cpu_masks"].asCString()));
        }
      }

      builder.AddWorkers(workers);
      builder.AddWorkerCPUMasks(cpu_masks);
      builder.AddWorkerNumThreads(num_threads);
    }

    if (root["availability_check_interval_ms"].isInt()) {
      builder.AddAvailabilityCheckIntervalMs(
          root["availability_check_interval_ms"].asInt());
    }
  }

  // Runtime config
  {
    if (root["minimum_subgraph_size"].isInt()) {
      builder.AddMinimumSubgraphSize(root["minimum_subgraph_size"].asInt());
    }

    if (root["subgraph_preparation_type"].isString()) {
      builder.AddSubgraphPreparationType(FromString<SubgraphPreparationType>(
          root["subgraph_preparation_type"].asCString()));
    }

    if (root["cpu_masks"].isString()) {
      builder.AddCPUMask(
          FromString<CPUMaskFlag>(root["cpu_masks"].asCString()));
    }
  }

  auto profile_config_status = builder.Build();
  RETURN_IF_ERROR(profile_config_status.status());

  return new RuntimeConfig(profile_config_status.value());
}

absl::Status EngineRunner::Initialize(const Json::Value& root) {
  auto runtime_config = LoadRuntimeConfigs(root);
  RETURN_IF_ERROR(runtime_config.status());
  RETURN_IF_ERROR(LoadRunnerConfigs(root));

  engine_ = Engine::Create(*runtime_config.value()).release();
  runtime_config_ = runtime_config.value();

  if (!engine_) {
    return absl::InternalError("Failed to create engine");
  }
  return absl::OkStatus();
}

absl::Status EngineRunner::Run() {
  for (size_t i = 0; i < children_.size(); i++) {
    RETURN_IF_ERROR(children_[i]->Run());
  }
}
}  // namespace tool
}  // namespace band