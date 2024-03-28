/*
 * Copyright 2023 Seoul National University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef BAND_TOOL_BENCHMARK_CONFIG_H_
#define BAND_TOOL_BENCHMARK_CONFIG_H_

#include "band/config.h"

namespace band {
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

struct BenchmarkConfig {
  std::vector<ModelConfig> model_configs;
  std::string execution_mode;
  size_t running_time_ms = 60000;
  // TODO: add workload simulator
};
}  // namespace tool
}  // namespace band

#endif