#include "band/tool/engine_runner.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>

#include "absl/strings/str_format.h"
#include "band/config_builder.h"
#include "band/logger.h"
#include "band/model.h"
#include "band/profiler.h"
#include "band/tensor.h"
#include "band/time.h"
#include "band/tool/benchmark.h"
#include "engine_runner.h"

namespace band {
namespace tool {
EngineRunner::EngineRunner(BackendType target_backend)
    : target_backend_(target_backend) {}

EngineRunner::~EngineRunner() {
  if (runtime_config_) {
    delete runtime_config_;
  }
}

absl::Status EngineRunner::LoadBenchmarkConfigs(const Json::Value& root) {
  if (!json::Validate(root, {"execution_mode", "models"})) {
    return absl::InvalidArgumentError(
        "Please check if argument `execution_mode` and `models` are given");
  }

  json::AssignIfValid(benchmark_config_.execution_mode, root, "execution_mode");

  std::set<std::string> supported_execution_modes{"periodic", "stream"};
  if (supported_execution_modes.find(benchmark_config_.execution_mode) ==
      supported_execution_modes.end()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Please check if argument execution mode %s is valid",
                        benchmark_config_.execution_mode));
  }

  json::AssignIfValid(benchmark_config_.running_time_ms, root,
                      "running_time_ms");

  if (benchmark_config_.execution_mode == "periodic") {
    if (!json::AssignIfValid(benchmark_config_.period_ms, root, "period_ms") ||
        benchmark_config_.period_ms == 0) {
      return absl::InvalidArgumentError(
          "Please check if argument `period_ms` is given and >= 0");
    }
  }

  if (benchmark_config_.running_time_ms == 0) {
    return absl::InvalidArgumentError(
        "Please check if argument running_time_ms >= 0");
  }

  if (root["models"].size() == 0) {
    return absl::InvalidArgumentError("Please specify at list one model");
  }

  // Set Model Configurations
  for (int i = 0; i < root["models"].size(); ++i) {
    ModelConfig model;
    Json::Value model_json_value = root["models"][i];

    // Set model filepath.
    // Required for all cases.
    if (!json::AssignIfValid(model.path, model_json_value, "graph")) {
      return absl::InvalidArgumentError(
          "Please check if argument `graph` is given in the model configs");
    }

    // Set `period_ms`.
    // Required for `periodic` mode.
    if (benchmark_config_.execution_mode == "periodic") {
      if (!json::AssignIfValid(model.period_ms, model_json_value,
                               "period_ms") ||
          model.period_ms == 0) {
        return absl::InvalidArgumentError(
            "Please check if argument `period_ms` is given and >= 0");
      }
    }

    json::AssignIfValid(model.batch_size, model_json_value, "batch_size");
    json::AssignIfValid(model.worker_id, model_json_value, "worker_id");
    json::AssignIfValid(model.slo_us, model_json_value, "slo_us");
    json::AssignIfValid(model.slo_scale, model_json_value, "slo_scale");

    benchmark_config_.model_configs.push_back(model);
  }
  return absl::OkStatus();
}

absl::Status tool::EngineRunner::LoadRuntimeConfigs(const Json::Value& root) {
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

  if (!builder.IsValid()) {
    return absl::InvalidArgumentError("Invalid runtime config");
  }

  runtime_config_ = new RuntimeConfig(builder.Build());

  return absl::OkStatus();
}

// motivated from /tensorflow/lite/tools/benchmark
template <typename T, typename Distribution>
void CreateRandomTensorData(void* target_ptr, int num_elements,
                            Distribution distribution) {
  std::mt19937 random_engine;
  T* target_head = static_cast<T*>(target_ptr);
  std::generate_n(target_head, num_elements, [&]() {
    return static_cast<T>(distribution(random_engine));
  });
}

absl::Status EngineRunner::Initialize(const Json::Value& root) {
  RETURN_IF_ERROR(LoadBenchmarkConfigs(root));
  RETURN_IF_ERROR(LoadRuntimeConfigs(root));

  engine_ = Engine::Create(*runtime_config_);
  if (!engine_) {
    return absl::InternalError("Failed to create engine");
  }

  // load models
  for (auto& benchmark_model : benchmark_config_.model_configs) {
    ModelContext* engine = new ModelContext;

    {
      auto status =
          engine->model.FromPath(target_backend_, benchmark_model.path.c_str());
      if (!status.ok()) {
        return status;
      }
    }
    {
      auto status = engine_->RegisterModel(&engine->model);
      if (!status.ok()) {
        return status;
      }
    }

    const int model_id = engine->model.GetId();
    const auto input_indices = engine_->GetInputTensorIndices(model_id);
    const auto output_indices = engine_->GetOutputTensorIndices(model_id);

    for (int i = 0; i < benchmark_model.batch_size; i++) {
      // pre-allocate tensors
      Tensors inputs, outputs;
      for (int input_index : input_indices) {
        inputs.push_back(engine_->CreateTensor(model_id, input_index));
      }

      for (int output_index : output_indices) {
        outputs.push_back(engine_->CreateTensor(model_id, output_index));
      }

      engine->model_request_inputs.push_back(inputs);
      engine->model_request_outputs.push_back(outputs);
    }

    if (benchmark_model.slo_us <= 0 && benchmark_model.slo_scale > 0.f) {
      int64_t worst_us = 0;

      // calculate worst case latency
      for (int worker_id = 0; worker_id < engine_->GetNumWorkers();
           worker_id++) {
        worst_us = std::max(engine_->GetProfiled(engine_->GetLargestSubgraphKey(
                                model_id, worker_id)),
                            worst_us);
      }

      if (worst_us == 0) {
        std::cout << "Failed to get worst case latency for model "
                  << benchmark_model.path << std::endl;
        std::cout << "Please check if given planner types require profiling"
                  << std::endl;
      } else {
        benchmark_model.slo_us = worst_us * benchmark_model.slo_scale;
      }
    }

    engine->model_ids =
        std::vector<ModelId>(benchmark_model.batch_size, model_id);
    engine->request_options = std::vector<RequestOption>(
        benchmark_model.batch_size, benchmark_model.GetRequestOption());

    // pre-allocate random input tensor to feed in run-time
    Tensors inputs;
    for (int input_index : input_indices) {
      interface::ITensor* input_tensor =
          engine_->CreateTensor(model_id, input_index);
      // random value ranges borrowed from tensorflow/lite/tools/benchmark
      switch (input_tensor->GetType()) {
        case DataType::kUInt8:
          CreateRandomTensorData<uint8_t>(
              input_tensor->GetData(), input_tensor->GetNumElements(),
              std::uniform_int_distribution<int32_t>(0, 254));
          break;
        case DataType::kInt8:
          CreateRandomTensorData<int8_t>(
              input_tensor->GetData(), input_tensor->GetNumElements(),
              std::uniform_int_distribution<int32_t>(-127, 127));
          break;
        case DataType::kInt16:
          CreateRandomTensorData<int16_t>(
              input_tensor->GetData(), input_tensor->GetNumElements(),
              std::uniform_int_distribution<int16_t>(0, 99));
          break;
        case DataType::kInt32:
          CreateRandomTensorData<int32_t>(
              input_tensor->GetData(), input_tensor->GetNumElements(),
              std::uniform_int_distribution<int32_t>(0, 99));
          break;
        case DataType::kInt64:
          CreateRandomTensorData<int64_t>(
              input_tensor->GetData(), input_tensor->GetNumElements(),
              std::uniform_int_distribution<int64_t>(0, 99));
          break;
        case DataType::kFloat32:
          CreateRandomTensorData<float>(
              input_tensor->GetData(), input_tensor->GetNumElements(),
              std::uniform_real_distribution<float>(-0.5, 0.5));
          break;
        case DataType::kFloat64:
          CreateRandomTensorData<double>(
              input_tensor->GetData(), input_tensor->GetNumElements(),
              std::uniform_real_distribution<double>(-0.5, 0.5));
          break;

        default:
          break;
      }
      inputs.push_back(input_tensor);
    }
    engine->model_inputs = inputs;
    model_contexts_.push_back(engine);
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