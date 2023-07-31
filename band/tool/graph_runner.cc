#include "band/tool/graph_runner.h"

#include <random>

#include "absl/strings/str_format.h"
#include "band/engine.h"
#include "band/tool/engine_runner.h"
#include "band/tool/graph_context.h"
#include "graph_runner.h"

namespace band {
namespace tool {
GraphRunner::GraphRunner(BackendType target_backend,
                         EngineRunner& engine_runner)
    : target_backend_(target_backend), engine_runner_(engine_runner) {
  callback_id_ = engine_runner_.GetEngine().SetOnEndRequest(
      std::bind(&GraphRunner::OnJobFinished, this, std::placeholders::_1,
                std::placeholders::_2));
}

GraphRunner::~GraphRunner() {
  Join();
  engine_runner_.GetEngine().UnsetOnEndRequest(callback_id_);
}

absl::Status GraphRunner::Initialize(const Json::Value& root) {
  std::lock_guard<std::mutex> lock(mutex_);
  graph_contexts_.clear();
  job_id_to_graph_vertex_.clear();

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

  root_ = root;

  // create a single graph context and see if it is valid
  // execution mode requires multiple contexts (such as periodic)
  // will create multiple contexts later
  std::unique_ptr<GraphContext> temp_ctx =
      std::make_unique<GraphContext>(engine_runner_.GetEngine());
  RETURN_IF_ERROR(temp_ctx->Initialize(root, engine_runner_));
  graph_contexts_.emplace_back(std::move(temp_ctx));
  return absl::OkStatus();
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
    runner_safe_bool_.terminate();
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
  while (true) {
    if (runner_safe_bool_.wait()) {
      break;
    }
  }

  for (int model_index = 0; model_index < vertices_.size(); model_index++) {
    std::thread t(
        [this](ModelContext* model_context, const size_t period_us) {
          while (true) {
            if (!model_context->PrepareInput().ok()) {
              BAND_LOG_PROD(BAND_LOG_WARNING, "Failed to prepare input");
              continue;
            }

            size_t id = model_context->profiler_.BeginEvent();
            auto status = engine_.RequestSync(
                model_context->model_ids, model_context->request_options,
                model_context->model_request_inputs,
                model_context->model_request_outputs);
            model_context->profiler_.EndEvent(id, status);

            if (kill_app_) return;

            size_t elapsed_us =
                model_context->profiler_
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
  // get the only graph context
  size_t graph_index = 0;
  GraphContext* graph_context = graph_contexts_[0].get();
  graph_index = profiler_.BeginEvent();
  // prepare input for all vertices
  graph_context->InitializeExecutionContext();

  while (true) {
    // wait until all vertices are finished
    if (runner_safe_bool_.wait()) {
      break;
    }

    if (graph_context->IsFinished()) {
      profiler_.EndEvent(graph_index);
      graph_index = profiler_.BeginEvent();
      graph_context->InitializeExecutionContext();
    }

    // get next vertices to run
    auto vertices = graph_context->GetNextVertices();
    // aggregate model ids and request options
    std::vector<ModelId> model_ids;
    std::vector<RequestOption> request_options;
    std::vector<Tensors> model_request_inputs;
    std::vector<Tensors> model_request_outputs;

    for (auto& vertex : vertices) {
      model_ids.insert(model_ids.end(), vertex->model_ids.begin(),
                       vertex->model_ids.end());
      request_options.insert(request_options.end(),
                             vertex->request_options.begin(),
                             vertex->request_options.end());
      model_request_inputs.insert(model_request_inputs.end(),
                                  vertex->model_request_inputs.begin(),
                                  vertex->model_request_inputs.end());
      model_request_outputs.insert(model_request_outputs.end(),
                                   vertex->model_request_outputs.begin(),
                                   vertex->model_request_outputs.end());
    }

    size_t id = profiler_.BeginEvent();
    auto status =
        engine_.RequestSync(model_ids, request_options, model_request_inputs,
                            model_request_outputs);
  }
}

void GraphRunner::RunWorkload() { BAND_NOT_IMPLEMENTED; }

void GraphRunner::OnJobFinished(JobId job_id, absl::Status status) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = job_id_to_graph_vertex_.find(job_id);
  if (it == job_id_to_graph_vertex_.end()) {
    BAND_LOG_PROD(BAND_LOG_WARNING,
                  "Job id %d is not found in job_id_to_graph_vertex_", job_id);
    return;
  }
  it->second.first->OnVertexFinished(it->second.second);
  job_id_to_graph_vertex_.erase(it);

  // notify runner thread if all vertices are finished for a single graph
  runner_safe_bool_.notify();
}

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

  auto print_profiler_ = [](const std::string& prefix,
                            const BenchmarkProfiler& profiler_,
                            const Vertex* model_config = nullptr) {
    const double batch_size = model_config ? model_config->batch_size : 1;
    double average_ms =
        (profiler_.GetAverageElapsedTime<std::chrono::milliseconds>() /
         batch_size);
    double average_fps = 1000 / average_ms;

    PrintHeader("Result - " + prefix);
    PrintLine("# Processed requests", profiler_.GetNumEvents() * batch_size, 1);
    PrintLine("Avg. Latency (ms)", average_ms, 1);
    PrintLine("Avg. FPS", average_fps, 1);
    PrintLine("Total # requests", profiler_.GetNumEvents() * batch_size, 1);
    PrintLine("Total # canceled requests",
              profiler_.GetNumCanceledEvents() * batch_size, 1);

    if (model_config && model_config->slo_us > 0) {
      double slo_satisfactory_count = 0;
      for (size_t i = 0; i < profiler_.GetNumEvents(); i++) {
        if (!profiler_.IsEventCanceled(i) &&
            profiler_.GetElapsedTimeAt<std::chrono::microseconds>(i) <
                model_config->slo_us) {
          slo_satisfactory_count++;
        }
      }

      double num_events =
          profiler_.GetNumEvents() - profiler_.GetNumCanceledEvents();

      PrintLine("SLO Satisfactory Rate (%)",
                slo_satisfactory_count / num_events * 100, 1);
    }
  };

  for (size_t model_index = 0; model_index < vertices_.size(); model_index++) {
    auto& model_context = vertices_[model_index];
    auto& model_config = config_.model_configs[model_index];

    if (model_context->profiler_.GetNumEvents() > 0) {
      print_profiler_(
          model_context->model.GetBackendModel(target_backend_)->GetPath(),
          model_context->profiler_, &model_config);
    }
  }

  return absl::OkStatus();
}
}  // namespace tool
}  // namespace band