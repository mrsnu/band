#include "band/tool/graph_runner.h"

#include <random>

#include "absl/strings/str_format.h"
#include "band/engine.h"
#include "band/tool/engine_runner.h"
#include "band/tool/graph_context.h"

namespace band {
namespace tool {

absl::Status GraphRunner::Initialize(const Json::Value& root) {
  if (!json::Validate(root, {"execution_mode", "vertices"})) {
    return absl::InvalidArgumentError(
        "Please check if argument `execution_mode` and `vertices` are given");
  }

  json::AssignIfValid(execution_mode_, root, "execution_mode");

  std::set<std::string> supported_execution_modes{"periodic", "stream"};
  if (supported_execution_modes.find(execution_mode_) ==
      supported_execution_modes.end()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Please check if argument execution mode %s is valid",
                        execution_mode_));
  }

  if (execution_mode_ == "periodic") {
    if (!json::AssignIfValid(period_ms_, root, "period_ms") ||
        period_ms_ == 0) {
      return absl::InvalidArgumentError(
          "Please check if argument `period_ms` is given and >= 0");
    }
  }

  if (json::AssignIfValid(slo_ms_, root, "slo_ms")) {
    if (slo_ms_ <= 0) {
      return absl::InvalidArgumentError("Please check if argument slo_ms >= 0");
    }
  }
  if (json::AssignIfValid(slo_scale_, root, "slo_scale")) {
    if (slo_scale_ <= 0) {
      return absl::InvalidArgumentError(
          "Please check if argument slo_scale >= 0");
    }
  }

  if (root["vertices"].size() == 0) {
    return absl::InvalidArgumentError("Please specify at list one model");
  }

  std::unique_ptr<GraphContext> temp_ctx =
      std::make_unique<GraphContext>(engine_runner_.GetEngine());
  return temp_ctx->Initialize(root, engine_runner_);
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
  if (execution_mode_ == "periodic") {
    RunPeriodic();
  } else if (execution_mode_ == "stream") {
    RunStream();
  } else if (execution_mode_ == "workload") {
    RunWorkload();
  }
}

void GraphRunner::RunPeriodic() {
  for (int model_index = 0; model_index < vertices_.size(); model_index++) {
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
        vertices_[model_index],
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

    for (auto model_context : vertices_) {
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
  PrintLine("Execution mode", execution_mode_, 1);
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

  for (size_t model_index = 0; model_index < vertices_.size(); model_index++) {
    auto& model_context = vertices_[model_index];
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
}  // namespace band