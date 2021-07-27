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

#include <json/json.h>
#include <fstream>

#include "tensorflow/lite/tools/logging.h"

namespace tflite {

// Note : program aborts when asX fails below
// e.g., asInt, asCString, ...
TfLiteStatus ParseRuntimeConfigFromJson(std::string json_fname,
                                        RuntimeConfig& runtime_config) {
  std::ifstream config(json_fname, std::ifstream::binary);

  Json::Value root;
  config >> root;

  if (!root.isObject()) {
    TFLITE_LOG(ERROR) << "Check the json config file format.";
    return kTfLiteError;
  }

  TFLITE_LOG(INFO) << root;

  if (ValidateJsonConfig(root, {"log_path", "planner_types"}) != kTfLiteOk) {
    return kTfLiteError;
  }

  auto& interpreter_config = runtime_config.interpreter_config;
  auto& planner_config = runtime_config.planner_config;
  auto& worker_config = runtime_config.worker_config;

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

  // Set Planner configs
  // 1. Log path
  planner_config.log_path = root["log_path"].asString();
  // 2. Scheduling window size
  if (!root["schedule_window_size"].isNull()) {
    planner_config.schedule_window_size = root["schedule_window_size"].asInt();
    if (planner_config.schedule_window_size <= 0) {
      TFLITE_LOG(ERROR) << "Make sure `schedule_window_size` > 0.";
      return kTfLiteError;
    }
  }
  // 3. Planner type
  for (int i = 0; i < root["schedulers"].size(); ++i) {
    int scheduler_id = root["schedulers"][i].asInt();
    if (scheduler_id < kFixedDevice || scheduler_id >= kNumSchedulerTypes) {
      TFLITE_LOG(ERROR) << "Wrong `schedulers` argument is given.";
      return kTfLiteError;
    }
    planner_config.schedulers.push_back(static_cast<TfLiteSchedulerType>(scheduler_id));
  }

  // Set Worker configs
  // 1. worker CPU masks
  if (!root["worker_cpu_masks"].isNull()) {
    for (auto const& key : root["worker_cpu_masks"].getMemberNames()) {
      size_t device_id = TfLiteDeviceGetFlag(key.c_str());
      impl::TfLiteCPUMaskFlags flag =
          impl::TfLiteCPUMaskGetMask(root["worker_cpu_masks"][key].asCString());
      if (device_id < kTfLiteNumDevices) {
        worker_config.cpu_masks[device_id] = flag;
      }
    }
  }
  for (auto device_id = 0; device_id < kTfLiteNumDevices; ++device_id) {
    if (worker_config.cpu_masks[device_id] == impl::kTfLiteNumCpuMasks) {
      worker_config.cpu_masks[device_id] = interpreter_config.cpu_masks;
    }
  }
  // 2. Allow worksteal
  if (!root["allow_work_steal"].isNull()) {
    worker_config.allow_worksteal = root["allow_work_steal"].asBool();
  }

  return kTfLiteOk;
}

TfLiteStatus ValidateJsonConfig(const Json::Value& json_config,
                                std::vector<std::string> keys) {
  for (auto key : keys) {
    if (json_config[key].isNull()) {
      TFLITE_LOG(ERROR) << "Please check if the argument `"
                        << key
                        << "` is given in the config file.";
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

}  // namespace tflite
