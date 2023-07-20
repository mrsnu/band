#include "band/tool/graph_runner.h"

#include <random>

#include "absl/strings/str_format.h"
#include "band/tool/engine_runner.h"

namespace band {
namespace tool {

GraphRunner::~GraphRunner() {
  for (auto model_context : model_contexts_) {
    delete model_context;
  }
}

absl::Status GraphRunner::Initialize(const Json::Value& root) {
  if (!json::Validate(root, {"execution_mode", "vertices"})) {
    return absl::InvalidArgumentError(
        "Please check if argument `execution_mode` and `vertices` are given");
  }

  json::AssignIfValid(config_.execution_mode, root, "execution_mode");

  std::set<std::string> supported_execution_modes{"periodic", "stream"};
  if (supported_execution_modes.find(config_.execution_mode) ==
      supported_execution_modes.end()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Please check if argument execution mode %s is valid",
                        config_.execution_mode));
  }

  if (config_.execution_mode == "periodic") {
    if (!json::AssignIfValid(config_.period_ms, root, "period_ms") ||
        config_.period_ms == 0) {
      return absl::InvalidArgumentError(
          "Please check if argument `period_ms` is given and >= 0");
    }
  }

  if (json::AssignIfValid(config_.slo_ms, root, "slo_ms")) {
    if (config_.slo_ms <= 0) {
      return absl::InvalidArgumentError("Please check if argument slo_ms >= 0");
    }
  }
  if (json::AssignIfValid(config_.slo_scale, root, "slo_scale")) {
    if (config_.slo_scale <= 0) {
      return absl::InvalidArgumentError(
          "Please check if argument slo_scale >= 0");
    }
  }

  if (root["vertices"].size() == 0) {
    return absl::InvalidArgumentError("Please specify at list one model");
  }

  for (auto vertex_key : root["vertices"].getMemberNames()) {
    if (!root["vertices"][vertex_key].isObject()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Please check if model config for model %s is "
                          "given",
                          vertex_key.c_str()));
    }

    if (!json::Validate(root["vertices"][vertex_key], {"name"})) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Please check if model name for vertex %s is "
                          "given",
                          vertex_key.c_str()));
    }

    const std::string model_key =
        root["vertices"][vertex_key]["name"].asString();
    auto model = engine_runner_.GetModel(model_key);
    if (!model.ok()) {
      return model.status();
    }

    Vertex* vertex = new Vertex();
    vertex->path = model_key;
    vertex->model_id = model->GetId();
    if (json::AssignIfValid(vertex->batch_size, root["vertices"][vertex_key],
                            "batch_size")) {
      if (vertex->batch_size <= 0) {
        return absl::InvalidArgumentError(
            "Please check if argument batch_size >= 0");
      }
    }
    if (json::AssignIfValid(vertex->worker_id, root["vertices"][vertex_key],
                            "worker_id")) {
      if ((vertex->worker_id < 0) ||
          (vertex->worker_id >= engine_runner_.GetEngine().GetNumWorkers())) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Please check if argument worker_id is valid (0 ~ %zu)",
            engine_runner_.GetEngine().GetNumWorkers() - 1));
      }
    }

    // Set Model Configurations
    for (int i = 0; i < root["vertices"].size(); ++i) {
      Vertex model;
      Json::Value model_json_value = root["vertices"][i];

      // Set model filepath.
      // Required for all cases.
      if (!json::AssignIfValid(model.path, model_json_value, "graph")) {
        return absl::InvalidArgumentError(
            "Please check if argument `graph` is given in the model configs");
      }

      // Set `period_ms`.
      // Required for `periodic` mode.
      if (config_.execution_mode == "periodic") {
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

      config_.model_configs.push_back(model);
    }

    // load models
    for (auto& benchmark_model : model_contexts_) {
      ModelContext* engine = new ModelContext;

      {
        auto status = engine->model.FromPath(target_backend_,
                                             benchmark_model.path.c_str());
        if (!status.ok()) {
          return status;
        }
      }
      {
        auto status = engine_.RegisterModel(&engine->model);
        if (!status.ok()) {
          return status;
        }
      }

      const int model_id = engine->model.GetId();
      const auto input_indices = engine_.GetInputTensorIndices(model_id);
      const auto output_indices = engine_.GetOutputTensorIndices(model_id);

      for (int i = 0; i < benchmark_model.batch_size; i++) {
        // pre-allocate tensors
        Tensors inputs, outputs;
        for (int input_index : input_indices) {
          inputs.push_back(engine_.CreateTensor(model_id, input_index));
        }

        for (int output_index : output_indices) {
          outputs.push_back(engine_.CreateTensor(model_id, output_index));
        }

        engine->model_request_inputs.push_back(inputs);
        engine->model_request_outputs.push_back(outputs);
      }

      if (benchmark_model.slo_us <= 0 && benchmark_model.slo_scale > 0.f) {
        int64_t worst_us = 0;

        // calculate worst case latency
        for (int worker_id = 0; worker_id < engine_.GetNumWorkers();
             worker_id++) {
          worst_us = std::max(engine_.GetProfiled(engine_.GetLargestSubgraphKey(
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
            engine_.CreateTensor(model_id, input_index);
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

    return absl::Status();
  }

  absl::Status GraphRunner::Run() {
    // return error if thread is running
    if (runner_thread_.joinable()) {
      return absl::InternalError("Benchmark thread is already running");
    }

    runner_thread_ = std::thread(&GraphRunner::RunInternal, this);
    return absl::OkStatus();
  }

  void GraphRunner::Join() {
    if (runner_thread_.joinable()) {
      runner_thread_.join();
    }
  }

  void GraphRunner::RunInternal() {
    if (config_.execution_mode == "periodic") {
      RunPeriodic();
    } else if (config_.execution_mode == "stream") {
      RunStream();
    } else if (config_.execution_mode == "workload") {
      RunWorkload();
    }
  }

  void GraphRunner::RunPeriodic() {
    for (int model_index = 0; model_index < model_contexts_.size();
         model_index++) {
      std::thread t(
          [this](ModelContext* model_context, const size_t period_us) {
            while (true) {
              if (!model_context->PrepareInput().ok()) {
                BAND_LOG_PROD(BAND_LOG_WARNING, "Failed to prepare input");
                continue;
              }

              size_t id = model_context->profiler.BeginEvent();
              auto status = engine_.RequestSync(
                  model_context->model_ids, model_context->request_options,
                  model_context->model_request_inputs,
                  model_context->model_request_outputs);
              model_context->profiler.EndEvent(id, status);

              if (kill_app_) return;

              size_t elapsed_us =
                  model_context->profiler
                      .GetElapsedTimeAt<std::chrono::microseconds>(id);

              if (elapsed_us < period_us) {
                std::this_thread::sleep_for(
                    std::chrono::microseconds(period_us - elapsed_us));
              }
            }
          },
          model_contexts_[model_index],
          config_.model_configs[model_index].period_ms * 1000);

      t.detach();
    }
    // wait for some time until we stop the benchmark
    std::this_thread::sleep_for(
        std::chrono::milliseconds(config_.running_time_ms));
    kill_app_ = true;
    engine_.WaitAll();
  }

  void GraphRunner::RunStream() {
    int run_duration_us = config_.running_time_ms * 1000;
    int64_t start = time::NowMicros();
    while (true) {
      std::vector<ModelId> model_ids;
      std::vector<RequestOption> request_options;
      std::vector<Tensors> inputs;
      std::vector<Tensors> outputs;

      for (auto model_context : model_contexts_) {
        if (!model_context->PrepareInput().ok()) {
          BAND_LOG_PROD(BAND_LOG_WARNING, "Failed to prepare input");
          continue;
        }

        model_ids.insert(model_ids.end(), model_context->model_ids.begin(),
                         model_context->model_ids.end());
        request_options.insert(request_options.end(),
                               model_context->request_options.begin(),
                               model_context->request_options.end());
        inputs.insert(inputs.end(), model_context->model_request_inputs.begin(),
                      model_context->model_request_inputs.end());
        outputs.insert(outputs.end(),
                       model_context->model_request_outputs.begin(),
                       model_context->model_request_outputs.end());
      }

      size_t id = global_profiler_.BeginEvent();
      auto status =
          engine_.RequestSync(model_ids, request_options, inputs, outputs);
      global_profiler_.EndEvent(id, status);
      int64_t current = time::NowMicros();
      if (current - start >= run_duration_us) break;
    }
  }

  void GraphRunner::RunWorkload() { BAND_NOT_IMPLEMENTED; }

  void PrintHeader(std::string key, size_t indent_level = 0) {
    std::cout << std::left << std::string(indent_level * 2, ' ') << "<" << key
              << ">" << std::endl;
  }

  template <typename T>
  void PrintLine(std::string key, const T& value, size_t indent_level = 0) {
    std::cout << std::left << std::string(indent_level * 2, ' ') << "[" << key
              << "] : " << std::right << value << std::endl;
  }

  absl::Status GraphRunner::LogResults(size_t instance_id) {
    const std::string header =
        "--\t\t\t Instance " + std::to_string(instance_id) + " \t\t\t--";
    size_t length = header.size();
    std::cout << std::setfill('-') << std::setw(length) << std::fixed;
    std::cout << header << std::endl;

    PrintHeader("Option");
    PrintLine("Execution mode", config_.execution_mode, 1);
    PrintLine("Running time (ms)", config_.running_time_ms, 1);
    for (auto& scheduler : runtime_config_->planner_config.schedulers) {
      PrintLine("Scheduler", ToString(scheduler), 1);
    }

    PrintHeader("Model");
    for (auto& model_config : config_.model_configs) {
      PrintHeader(model_config.path, 1);
      PrintLine("Batch size", model_config.batch_size, 2);
      PrintLine("Request period (ms)", model_config.period_ms, 2);
      PrintLine("SLO (us)", model_config.slo_us, 2);
      PrintLine("SLO scale", model_config.slo_scale, 2);
    }

    auto print_profiler = [](const std::string& prefix,
                             const BenchmarkProfiler& profiler,
                             const Vertex* model_config = nullptr) {
      const double batch_size = model_config ? model_config->batch_size : 1;
      double average_ms =
          (profiler.GetAverageElapsedTime<std::chrono::milliseconds>() /
           batch_size);
      double average_fps = 1000 / average_ms;

      PrintHeader("Result - " + prefix);
      PrintLine("# Processed requests", profiler.GetNumEvents() * batch_size,
                1);
      PrintLine("Avg. Latency (ms)", average_ms, 1);
      PrintLine("Avg. FPS", average_fps, 1);
      PrintLine("Total # requests", profiler.GetNumEvents() * batch_size, 1);
      PrintLine("Total # canceled requests",
                profiler.GetNumCanceledEvents() * batch_size, 1);

      if (model_config && model_config->slo_us > 0) {
        double slo_satisfactory_count = 0;
        for (size_t i = 0; i < profiler.GetNumEvents(); i++) {
          if (!profiler.IsEventCanceled(i) &&
              profiler.GetElapsedTimeAt<std::chrono::microseconds>(i) <
                  model_config->slo_us) {
            slo_satisfactory_count++;
          }
        }

        double num_events =
            profiler.GetNumEvents() - profiler.GetNumCanceledEvents();

        PrintLine("SLO Satisfactory Rate (%)",
                  slo_satisfactory_count / num_events * 100, 1);
      }
    };

    for (size_t model_index = 0; model_index < model_contexts_.size();
         model_index++) {
      auto& model_context = model_contexts_[model_index];
      auto& model_config = config_.model_configs[model_index];

      if (model_context->profiler.GetNumEvents() > 0) {
        print_profiler(
            model_context->model.GetBackendModel(target_backend_)->GetPath(),
            model_context->profiler, &model_config);
      }
    }

    return absl::OkStatus();
  }
}
}  // namespace tool
}  // namespace band