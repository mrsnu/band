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

  for (auto& engine_runner : children_) {
    engine_runner->Join();
  }

  return LogResults();
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
      [this](const Json::Value& engine_runner_config) -> absl::Status {
    std::unique_ptr<EngineRunner> engine_runner =
        std::make_unique<EngineRunner>();
    RETURN_IF_ERROR(engine_runner->Initialize(engine_runner_config));
    children_.emplace_back(engine_runner.release());
    return absl::OkStatus();
  };

  Json::Value json_config = json::LoadFromFile(argv[1]);

  if (json_config.isArray()) {
    // iterate through all benchmark instances (app, benchmark instance config)
    for (int i = 0; i < json_config.size(); ++i) {
      RETURN_IF_ERROR(try_create_engine_runner(json_config[i]));
    }
  } else {
    RETURN_IF_ERROR(try_create_engine_runner(json_config));
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