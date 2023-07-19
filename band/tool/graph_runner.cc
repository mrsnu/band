#include "band/tool/graph_runner.h"

#include "graph_runner.h"

namespace band {
namespace tool {

GraphRunner::ModelContext::~ModelContext() {
  auto delete_tensors = [](Tensors& tensors) {
    for (auto t : tensors) {
      delete t;
    }
  };

  for (auto request_inputs : model_request_inputs) {
    delete_tensors(request_inputs);
  }

  for (auto request_outputs : model_request_outputs) {
    delete_tensors(request_outputs);
  }

  delete_tensors(model_inputs);
}

absl::Status GraphRunner::ModelContext::PrepareInput() {
  for (int batch_index = 0; batch_index < model_request_inputs.size();
       batch_index++) {
    for (int input_index = 0; input_index < model_inputs.size();
         input_index++) {
      auto status =
          model_request_inputs[batch_index][input_index]->CopyDataFrom(
              model_inputs[input_index]);
      if (!status.ok()) {
        return status;
      }
    }
  }
  return absl::OkStatus();
}

GraphRunner::~GraphRunner() {
  for (auto model_context : model_contexts_) {
    delete model_context;
  }
}

absl::Status GraphRunner::Initialize(const Json::Value& root) {
  RETURN_IF_ERROR(LoadBenchmarkConfigs(root));

  // load models
  for (auto& benchmark_model : model_contexts_) {
    ModelContext* engine = new ModelContext;

    {
      auto status = engine->model.FromPath(target_backend_,
                                           benchmark_model->path.c_str());
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

    for (int i = 0; i < benchmark_model->batch_size; i++) {
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

    if (benchmark_model->slo_us <= 0 && benchmark_model->slo_scale > 0.f) {
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
                  << benchmark_model->path << std::endl;
        std::cout << "Please check if given planner types require profiling"
                  << std::endl;
      } else {
        benchmark_model->slo_us = worst_us * benchmark_model->slo_scale;
      }
    }

    engine->model_ids =
        std::vector<ModelId>(benchmark_model->batch_size, model_id);
    engine->request_options = std::vector<RequestOption>(
        benchmark_model->batch_size, benchmark_model->GetRequestOption());

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
                           const ModelConfig* model_config = nullptr) {
    const double batch_size = model_config ? model_config->batch_size : 1;
    double average_ms =
        (profiler.GetAverageElapsedTime<std::chrono::milliseconds>() /
         batch_size);
    double average_fps = 1000 / average_ms;

    PrintHeader("Result - " + prefix);
    PrintLine("# Processed requests", profiler.GetNumEvents() * batch_size, 1);
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

}  // namespace tool
}  // namespace band