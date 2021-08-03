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

#include <string>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/cpu.h"
#include "tensorflow/lite/util.h"

namespace tflite {
struct ProfileConfig {
  int num_warmups = 3;
  int num_runs = 50;
};

struct InterpreterConfig {
  std::string profile_data_path;
  ProfileConfig profile_config;
  float profile_smoothing_factor = 0.1;
  impl::TfLiteCPUMaskFlags cpu_masks = impl::kTfLiteAll;
};

struct PlannerConfig {
  TfLitePlannerType planner_type = kFixedDevice;
  std::string log_path;
  int schedule_window_size = INT_MAX;
};

struct WorkerConfig {
  WorkerConfig() {
    for (int i = 0; i < kTfLiteNumDevices; i++) {
      cpu_masks[i] = impl::kTfLiteNumCpuMasks;
    }
  }
  impl::TfLiteCPUMaskFlags cpu_masks[kTfLiteNumDevices];
  bool allow_worksteal = false;
  int32_t availability_check_interval_ms = 30000;
};


struct RuntimeConfig {
  InterpreterConfig interpreter_config;
  PlannerConfig planner_config;
  WorkerConfig worker_config;
};

// Parse runtime config from a json file path
TfLiteStatus ParseRuntimeConfigFromJson(std::string json_fname,
                                        RuntimeConfig& runtime_config);

// Check if the keys exist in the config
TfLiteStatus ValidateJsonConfig(const Json::Value& json_config,
                                std::vector<std::string> keys);
}  // namespace tflite
#endif  // TENSORFLOW_LITE_CONFIG_H_
