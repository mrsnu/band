#ifndef BAND_TOOL_BENCHMARK_CONFIG_H_
#define BAND_TOOL_BENCHMARK_CONFIG_H_

#include "band/config.h"

namespace band {
namespace tool {
struct ModelConfig {
  std::string path;
  size_t batch_size = 1;
  int worker_id = -1;
  size_t model_id;

  const RequestOption GetRequestOption() const {
    RequestOption option = RequestOption::GetDefaultOption();
    if (worker_id >= 0) {
      option.target_worker = worker_id;
    }
    return option;
  }
};

struct GraphRunnerConfig {
  std::vector<ModelConfig> model_configs;
  std::vector<std::pair<size_t /* from */, size_t /* to */>> edges;
  std::string execution_mode;
  size_t period_ms;
  int slo_us = -1;
  float slo_scale = -1.f;
};

// global config for engine runner
struct EngineRunnerConfig {
  size_t running_time_ms = 60000;
};
}  // namespace tool
}  // namespace band

#endif