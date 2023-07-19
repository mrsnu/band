#include "band/tool/benchmark.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>

#include "band/config_builder.h"
#include "band/logger.h"
#include "band/model.h"
#include "band/profiler.h"
#include "band/tensor.h"
#include "band/time.h"
#include "band/tool/engine_runner.h"
#include "benchmark.h"

namespace band {
namespace tool {

absl::Status Benchmark::Run() {
  for (size_t i = 0; i < children_.size(); i++) {
    RETURN_IF_ERROR(children_[i]->Run());
  }

  for (auto engine_runner : children_) {
    engine_runner->Join();
  }

  return LogResults();
}

Benchmark::ModelContext::~ModelContext() {
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

absl::Status Benchmark::ModelContext::PrepareInput() {
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

absl::Status Benchmark::Initialize(int argc, const char** argv) {
  if (argc < 2) {
    std::cout << "Usage:\n\tbenchmark <config-json-path> [<verbosity> = "
                 "default value: WARNING]"
              << std::endl;
    std::cout << "List of valid verbosity levels:" << std::endl;
    for (int i = 0; i < BAND_LOG_NUM_SEVERITIES; i++) {
      std::cout << "\t" << i << " : "
                << band::Logger::GetSeverityName(static_cast<LogSeverity>(i))
                << std::endl;
    }

    return absl::InvalidArgumentError("Invalid argument.");
  }
  if (argc >= 3) {
    band::Logger::SetVerbosity(atoi(argv[2]));
  } else {
    band::Logger::SetVerbosity(BAND_LOG_WARNING);
  }

  auto try_create_engine_runner =
      [this](const Json::Value& engine_runner_config)
      -> absl::StatusOr<EngineRunner*> {
    EngineRunner* engine_runner = new EngineRunner();
    absl::Status status = engine_runner->Initialize(engine_runner_config);
    if (!status.ok()) {
      delete engine_runner;
      return status;
    }
    children_.push_back(engine_runner);
    return absl::OkStatus();
  };

  Json::Value json_config = json::LoadFromFile(argv[1]);

  if (json_config.isArray()) {
    // iterate through all benchmark instances (app, benchmark instance config)
    for (int i = 0; i < json_config.size(); ++i) {
      try_create_engine_runner(json_config[i]);
    }
  } else {
    try_create_engine_runner(json_config);
  }

  return absl::OkStatus();
}

absl::Status Benchmark::LogResults() {
  const std::string header = "--\t\t Band Benchmark Tool \t\t--";
  size_t length = header.size();
  std::cout << std::setfill('-') << std::setw(length) << std::fixed;
  std::cout << header << std::endl;

  return IRunner::LogResults(0);
}

}  // namespace tool
}  // namespace band