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
#include "tensorflow/lite/config.h"
#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/minimal_logging.h"

#include <json/json.h>

#include <fstream>

namespace tflite {

// Note : program aborts when asX fails below
// e.g., asInt, asCString, ...
TfLiteStatus ParseRuntimeConfigFromJsonObject(const Json::Value& root,
                                              RuntimeConfig& runtime_config,
                                              ErrorReporter* error_reporter) {
  if (ValidateJsonConfig(root, {"log_path", "schedulers"}, error_reporter) !=
      kTfLiteOk) {
    return kTfLiteError;
  }

  auto& interpreter_config = runtime_config.interpreter_config;
  auto& planner_config = runtime_config.planner_config;
  auto& worker_config = runtime_config.worker_config;
  auto& resource_config = runtime_config.resource_config;

  // Set Interpreter Configs
  // 1. CPU mask
  if (!root["cpu_masks"].isNull()) {
    interpreter_config.cpu_masks =
        impl::TfLiteCPUMaskGetMask(root["cpu_masks"].asCString());
  }
  // 2. Dynamic profile config
  if (!root["profile_smoothing_factor"].isNull()) {
    interpreter_config.profile_smoothing_factor =
        root["profile_smoothing_factor"].asFloat();
  }
  // 3. File path to profile data
  if (!root["model_profile"].isNull()) {
    interpreter_config.profile_data_path = root["model_profile"].asString();
  }
  // 4. Number of threads
  if (!root["num_threads"].isNull()) {
    interpreter_config.num_threads = root["num_threads"].asInt();
  }
  // 5. Profile config
  if (!root["profile_online"].isNull()) {
    interpreter_config.profile_config.online = root["profile_online"].asBool();
  }
  if (!root["profile_warmup_runs"].isNull()) {
    interpreter_config.profile_config.num_warmups =
        root["profile_warmup_runs"].asInt();
  }
  if (!root["profile_num_runs"].isNull()) {
    interpreter_config.profile_config.num_runs =
        root["profile_num_runs"].asInt();
  }
  if (!root["profile_copy_computation_ratio"].isNull()) {
    interpreter_config.copy_computation_ratio =
        root["profile_copy_computation_ratio"].asInt();
  }
  // 6. Subgraph preparation type
  if (!root["subgraph_preparation_type"].isNull()) {
    interpreter_config.subgraph_preparation_type =
        root["subgraph_preparation_type"].asString();
  }
  // 7. Minimum subgraph size
  if (!root["minimum_subgraph_size"].isNull()) {
    interpreter_config.minimum_subgraph_size =
        root["minimum_subgraph_size"].asInt();
  }

  // Set Planner configs
  // 1. Log path
  planner_config.log_path = root["log_path"].asString();
  // 2. Scheduling window size
  if (!root["schedule_window_size"].isNull()) {
    planner_config.schedule_window_size = root["schedule_window_size"].asInt();
    TF_LITE_ENSURE(error_reporter, planner_config.schedule_window_size > 0);
  }
  // 3. Planner type
  for (int i = 0; i < root["schedulers"].size(); ++i) {
    int scheduler_id = root["schedulers"][i].asInt();
    TF_LITE_ENSURE_MSG(
        error_reporter,
        scheduler_id >= kFixedDevice && scheduler_id < kNumSchedulerTypes,
        "Wrong `schedulers` argument is given.");
    planner_config.schedulers.push_back(
        static_cast<TfLiteSchedulerType>(scheduler_id));
  }
  // 4. Planner CPU masks
  if (!root["planner_cpu_masks"].isNull()) {
    planner_config.cpu_masks =
        impl::TfLiteCPUMaskGetMask(root["planner_cpu_masks"].asCString());
  } else {
    planner_config.cpu_masks = interpreter_config.cpu_masks;
  }

  std::vector<bool> found_default_worker(kTfLiteNumDevices, false);
  // Set Worker configs
  if (!root["workers"].isNull()) {
    for (int i = 0; i < root["workers"].size(); ++i) {
      auto worker_config_json = root["workers"][i];

      TF_LITE_ENSURE_MSG(
          error_reporter, !worker_config_json["device"].isNull(),
          "Please check if argument `device` is given in the worker configs.");

      TfLiteDeviceFlags device_flag =
          TfLiteDeviceGetFlag(worker_config_json["device"].asCString());
      TF_LITE_ENSURE_FORMATTED_MSG(error_reporter,
                                   device_flag != kTfLiteNumDevices,
                                   "Wrong `device` argument is given. %s",
                                   worker_config_json["device"].asCString());

      int worker_id = device_flag;
      // Add additional device worker
      if (found_default_worker[device_flag]) {
        worker_id = worker_config.workers.size();
        worker_config.workers.push_back(device_flag);
        worker_config.cpu_masks.push_back(impl::kTfLiteNumCpuMasks);
        worker_config.num_threads.push_back(0);
        interpreter_config.profile_config.copy_computation_ratio.push_back(0);
      } else {
        found_default_worker[device_flag] = true;
      }

      // 1. worker CPU masks
      if (!worker_config_json["cpu_masks"].isNull()) {
        impl::TfLiteCPUMaskFlags flag = impl::TfLiteCPUMaskGetMask(
            worker_config_json["cpu_masks"].asCString());
        worker_config.cpu_masks[worker_id] = flag;
      }

      // 2. worker num threads
      if (!worker_config_json["num_threads"].isNull()) {
        worker_config.num_threads[worker_id] =
            worker_config_json["num_threads"].asInt();
      }

      // Copy/computation ratio for profiling
      if (!worker_config_json["profile_copy_computation_ratio"].isNull()) {
        interpreter_config.profile_config.copy_computation_ratio[worker_id] =
            worker_config_json["profile_copy_computation_ratio"].asInt();
      }
    }
  }

  // Update default values from interpreter
  for (auto worker_id = 0; worker_id < worker_config.workers.size();
       ++worker_id) {
    if (worker_config.cpu_masks[worker_id] == impl::kTfLiteNumCpuMasks) {
      worker_config.cpu_masks[worker_id] = interpreter_config.cpu_masks;
    }
    if (worker_config.num_threads[worker_id] == 0) {
      worker_config.num_threads[worker_id] = interpreter_config.num_threads;
    }
    if (interpreter_config.profile_config.copy_computation_ratio[worker_id] ==
        0) {
      interpreter_config.profile_config.copy_computation_ratio[worker_id] =
          interpreter_config.copy_computation_ratio;
    }
  }

  // 3. Allow worksteal
  if (!root["allow_work_steal"].isNull()) {
    worker_config.allow_worksteal = root["allow_work_steal"].asBool();
  }
  // 3. availability_check_interval_ms
  if (!root["availability_check_interval_ms"].isNull()) {
    worker_config.availability_check_interval_ms =
        root["availability_check_interval_ms"].asInt();
  }
  if (!root["offloading_target"].isNull()) {
    worker_config.offloading_target = root["offloading_target"].asCString();
  }

  if (!root["offloading_data_size"].isNull()) {
    worker_config.offloading_data_size = root["offloading_data_size"].asInt();
  }

  // Set Resource configs
  // 1. Temperature log path
  resource_config.temperature_log_path = root["temperature_log_path"].asString();
  if (!root["resources"].isNull()) {
    for (int i = 0; i < root["resources"].size(); ++i) {
      auto resource_config_json = root["resources"][i];
      TF_LITE_ENSURE_MSG(
          error_reporter, !resource_config_json["device"].isNull(),
          "Please check if argument `device` is given in the resource configs.");
      std::string device = resource_config_json["device"].asCString();
      // 1. worker temperature zone path 
      if (!resource_config_json["tz_path"].isNull()) {
        resource_config.tz_path.emplace(device, resource_config_json["tz_path"].asCString());
      }
      // 2. worker frequency path
      if (!resource_config_json["freq_path"].isNull()) {
        resource_config.freq_path.emplace(device, resource_config_json["freq_path"].asCString());
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus ValidateJsonConfig(const Json::Value& json_config,
                                std::vector<std::string> keys,
                                ErrorReporter* error_reporter) {
  for (auto key : keys) {
    TF_LITE_ENSURE_FORMATTED_MSG(
        error_reporter, !json_config[key].isNull(),
        "Please check if the argument %s is given in the config file.",
        key.c_str());
  }
  return kTfLiteOk;
}

TfLiteStatus ParseRuntimeConfigFromJson(std::string json_fname,
                                        RuntimeConfig & runtime_config,
                                        ErrorReporter * error_reporter) {
  std::ifstream config(json_fname, std::ifstream::binary);
  TF_LITE_ENSURE(error_reporter, config.is_open());

  Json::Value root;
  config >> root;

  TF_LITE_ENSURE(error_reporter, root.isObject());
  TFLITE_LOG_INTERNAL(TFLITE_LOG_INFO, "Runtime config %s",
                      root.toStyledString().c_str());

  return ParseRuntimeConfigFromJsonObject(root, runtime_config, error_reporter);
}

TfLiteStatus ParseRuntimeConfigFromJson(
    const void* buffer, size_t buffer_length, RuntimeConfig& runtime_config,
    ErrorReporter* error_reporter) {
  Json::Reader reader;
  Json::Value root;

  TF_LITE_ENSURE(
      error_reporter,
      reader.parse(static_cast<const char*>(buffer),
                   static_cast<const char*>(buffer) + buffer_length, root));
  TF_LITE_ENSURE(error_reporter, root.isObject());

  return ParseRuntimeConfigFromJsonObject(root, runtime_config, error_reporter);
}

}  // namespace tflite
