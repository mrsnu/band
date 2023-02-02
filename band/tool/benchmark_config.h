#ifndef BAND_TOOL_BENCHMARK_CONFIG_H_
#define BAND_TOOL_BENCHMARK_CONFIG_H_

#include "band/config.h"

namespace Band {
namespace tool {

struct ModelConfig {
  /* mendatory */
  std::string path;
  size_t batch_size = 1;
  /* optional */
  size_t period_ms;  // for periodic requests
  int worker_id = -1;
  int slo_us = -1;
  float slo_scale = -1.f;

  const BandRequestOption GetRequestOption() const {
    BandRequestOption option = BandGetDefaultRequestOption();
    if (worker_id >= 0) {
      option.target_worker = worker_id;
    }
    if (slo_us >= 0) {
      option.slo_us = slo_us;
    }
    if (slo_scale >= 0) {
      option.slo_scale = slo_scale;
    }
    return option;
  }
};

struct BenchmarkConfig {
  std::vector<ModelConfig> model_configs;
  std::string execution_mode;
  size_t running_time_ms = 60000;
  // TODO: add workload simulator
};
}  // namespace tool
}  // namespace Band

#endif