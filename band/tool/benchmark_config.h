#ifndef BAND_TOOL_BENCHMARK_CONFIG_H_
#define BAND_TOOL_BENCHMARK_CONFIG_H_

#include "band/config.h"

namespace band {
namespace tool {

/* BenchmarkConfig

- app_defs (ApplicationConfig) : graph-based DNN workloads to be
    executed in a single application
  - model_configs
    - name
    - path
    - batch_size (optional) : default 1
    - worker_id (optional) : only for fixed-worker mode to specify the worker
  - edges
    - (from, to): from and to are ModelConfig's name

- engine_instance (EngineInstanceConfig) : a set of applications to be executed
in a single engine
  - apps
    - name
    - period_ms (optional) : only for periodic execution mode
    - execution_mode
  - running_time_ms
  - log_path
  -

- share_engine : whether to share the engine among applications
-
*/

struct ModelConfig {
  /* mendatory */
  std::string path;
  size_t batch_size = 1;
  /* optional */
  size_t period_ms;  // for periodic requests
  int worker_id = -1;
  int slo_us = -1;
  float slo_scale = -1.f;

  const RequestOption GetRequestOption() const {
    RequestOption option = RequestOption::GetDefaultOption();
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

struct BenchmarkInstanceConfig {
  std::vector<ModelConfig> model_configs;
  std::string execution_mode;
  size_t running_time_ms = 60000;
  // TODO: add workload simulator
};
}  // namespace tool
}  // namespace band

#endif