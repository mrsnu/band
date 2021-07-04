/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_CONFIG_H_
#define TENSORFLOW_LITE_CONFIG_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/cpu.h"
#include "tensorflow/lite/tools/benchmark/benchmark_utils.h"

namespace tflite {
  struct ProfileConfig {
    int num_warmups = 3;
    int num_runs = 50;
  };

  struct InterpreterConfig {
    std::string profile_data_path;
    ProfileConfig profile_config;
    float profiling_smoothing_factor = 0.1;
    TfLiteCPUMaskFlags cpu_mask = impl::kTfLiteAll;
  };

  struct PlannerConfig {
    TfLitePlannerType planner_type;
    std::string log_path;
    int schedule_window_size = INT_MAX;
  };

  struct WorkerConfig {
    TfLiteCPUMaskFlags worker_cpu_masks[kTfLiteNumDevices];
    bool allow_worksteal = false;
  };

  struct BenchmarkConfig {
    std::string execution_mode;
    unsigned model_id_random_seed;
    int global_period_ms;
    int running_time_ms = 60000;
  };

  struct ModelConfig {
    std::string model_fname;
    int period_ms;
    int device = -1;
    int batch_size = 1;
    int64_t slo_us = -1;
    float slo_scale = -1.f;
  };

  struct ModelInformation {
    ModelInformation(std::vector<InputLayerInfo> input_layer_infos,
                     ModelConfig config)
      :input_layer_infos(input_layer_infos), config(config) {}
    std::vector<InputLayerInfo> input_layer_infos;
    std::vector<InputTensorData> input_tensor_data;
    ModelConfig config;
  };

  struct RuntimeConfig {
    InterpreterConfig interpreter_config;
    PlannerConfig planner_config;
    WorkerConfig worker_config;
    BenchmarkConfig benchmark_config;
    std::vector<ModelInformation> model_information;
  };

  TfLiteStatus ParseJsonFile(std::string json_fname,
                             RuntimeConfig* runtime_config);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_CONFIG_H_
