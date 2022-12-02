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
#include <unordered_map>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/cpu.h"
#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/util.h"

namespace tflite {
struct ProfileConfig {
  ProfileConfig() {
    copy_computation_ratio = std::vector<int>(kTfLiteNumDevices, 0);
  }
  bool online = true;
  int num_warmups = 3;
  int num_runs = 1;
  std::vector<int> copy_computation_ratio;
};

struct InterpreterConfig {
  std::string profile_data_path;
  ProfileConfig profile_config;
  int minimum_subgraph_size = 7;
  float profile_smoothing_factor = 0.1;
  std::string subgraph_preparation_type = "no_fallback_subgraph";
  impl::TfLiteCPUMaskFlags cpu_masks = impl::kTfLiteAll;
  int copy_computation_ratio = 1000;
  int num_threads = -1;
};

struct PlannerConfig {
  std::string log_path;
  int schedule_window_size = 5;
  std::vector<TfLiteSchedulerType> schedulers;
  impl::TfLiteCPUMaskFlags cpu_masks = impl::kTfLiteAll;

};

struct WorkerConfig {
  WorkerConfig() {
    // Add one default worker per device
    for (int i = 0; i < kTfLiteNumDevices; i++) {
      workers.push_back(static_cast<TfLiteDeviceFlags>(i));
    }
    cpu_masks = std::vector<impl::TfLiteCPUMaskFlags>(kTfLiteNumDevices,
                                                      impl::kTfLiteNumCpuMasks);
    num_threads = std::vector<int>(kTfLiteNumDevices, 0);
  }
  std::vector<TfLiteDeviceFlags> workers;
  std::vector<impl::TfLiteCPUMaskFlags> cpu_masks;
  std::vector<int> num_threads;
  bool allow_worksteal = false;
  int32_t availability_check_interval_ms = 30000;
  std::string offloading_target = "";
  std::int32_t offloading_data_size = 0; 
};

struct ResourceConfig {
  std::vector<std::string> tz_path;
  std::vector<std::string> freq_path;
  std::vector<thermal_t> threshold;
  std::vector<std::string> target_tz_path;
  std::vector<thermal_t> target_threshold;
  int32_t model_update_window_size = 750;
  std::string latency_model_param_path;
  std::string cloud_latency_model_param_path;
  std::string thermal_model_param_path;
  float weighted_ppt_config = 0.3;
  std::string rssi_path;
};

struct RuntimeConfig {
  InterpreterConfig interpreter_config;
  PlannerConfig planner_config;
  WorkerConfig worker_config;
  ResourceConfig resource_config;
};

class ErrorReporter;
// Parse runtime config from a json file path
TfLiteStatus ParseRuntimeConfigFromJsonObject(const Json::Value& root,
                                        RuntimeConfig& runtime_config,
                                        ErrorReporter* error_reporter = DefaultErrorReporter());

TfLiteStatus ParseRuntimeConfigFromJson(std::string json_fname,
                                        RuntimeConfig& runtime_config,
                                        ErrorReporter* error_reporter = DefaultErrorReporter());
                                        
TfLiteStatus ParseRuntimeConfigFromJson(const void* buffer, size_t buffer_length,
                                        RuntimeConfig& runtime_config,
                                        ErrorReporter* error_reporter = DefaultErrorReporter());

// Check if the keys exist in the config
TfLiteStatus ValidateJsonConfig(const Json::Value& json_config,
                                std::vector<std::string> keys,
                                ErrorReporter* error_reporter = DefaultErrorReporter());
}  // namespace tflite
#endif  // TENSORFLOW_LITE_CONFIG_H_
