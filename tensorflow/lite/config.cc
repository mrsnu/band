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

namespace tflite {

// Note : program aborts when asX fails below
// e.g., asInt, asCString, ...
TfLiteStatus ParseJsonFile(std::string json_fname,
                           RuntimeConfig* runtime_config,
                           BenchmarkConfig* benchmark_config) {
  std::ifstream config(json_fname, std::ifstream::binary);

  Json::Value root;
  config >> root;

  if (!root.isObject()) {
    TFLITE_LOG(ERROR) << "Please validate the json config file.";
    return kTfLiteError;
  }

  auto& interpreter_config = runtime_config->interpreter_config;
  auto& planner_config = runtime_config->planner_config;
  auto& worker_config = runtime_config->worker_config;

  // Set Runtime Configurations
  // Optional
  if (!root["cpu_masks"].isNull()) {
    interpreter_config.cpu_masks =
        impl::TfLiteCPUMaskGetMask(root["cpu_masks"].asCString());
  }
  if (!root["worker_cpu_masks"].isNull()) {
    for (auto const& key : root["worker_cpu_masks"].getMemberNames()) {
      size_t device_id = TfLiteDeviceGetFlag(key.c_str());
      impl::TfLiteCPUMaskFlags flag =
          impl::TfLiteCPUMaskGetMask(root["worker_cpu_masks"][key].asCString());
      if (device_id < kTfLiteNumDevices && flag != impl::kTfLiteAll) {
        worker_config.cpu_masks[device_id] = flag;
      }
    }
  }
  
  for (auto device = kTfLiteCPU; device < kTfLiteNumDevices; ++device) {
    if (worker_config.cpu_masks[device] == kTfLiteNumCpuMasks) {
      worker_config.cpu_masks[device] = interpreter_config.cpu_masks;
    }
  }


  if (!root["profile_smoothing_factor"].isNull()) {
    interpreter_config.profile_smoothing_factor =
      root["profile_smoothing_factor"].asFloat();
  }
  if (!root["model_profile"].isNull()) {
    interpreter_config.profile_data_path = root["model_profile"].asString();
  }
  if (!root["allow_work_steal"].isNull()) {
    worker_config.allow_work_steal = root["allow_work_steal"].asBool();
  }
  if (!root["schedule_window_size"].isNull()) {
    planner_config.schedule_window_size = root["schedule_window_size"].asInt();
    if (planner_config.schedule_window_size <= 0) {
      TFLITE_LOG(ERROR) << "Make sure `schedule_window_size` > 0.";
      return kTfLiteError;
    }
  }
  if (!root["global_period_ms"].isNull()) {
    planner_config.global_period_ms = root["global_period_ms"].asInt();
    if (planner_config.global_period_ms <= 0) {
      TFLITE_LOG(ERROR) << "Make sure `global_period_ms` > 0.";
      return kTfLiteError;
    }
  }


  // Required
  if (root["log_path"].isNull() ||
      root["planner"].isNull() ||
      root["execution_mode"].isNull() ||
      root["models"].isNull()) {
    TFLITE_LOG(ERROR) << "Please check if arguments `execution_mode`, "
                      << "`log_path`, `planner` and `models`"
                      << " are given in the config file.";
    return kTfLiteError;
  }

  planner_config.log_path = root["log_path"].asString();

  int planner_id = root["planner"].asInt();
  if (planner_id < kFixedDevice || planner_id >= kNumPlannerTypes) {
    TFLITE_LOG(ERROR) << "Wrong `planner` argument is given.";
    return kTfLiteError;
  }
  planner_config.planner_type = static_cast<TfLitePlannerType>(planner_id);

  if (benchmark_config) {
    benchmark_config->execution_mode = root["execution_mode"].asString();
    if (!root["running_time_ms"].isNull()) {
      benchmark_config->running_time_ms = root["running_time_ms"].asInt();
    }
    if (!root["model_id_random_seed"].isNull()) {
      benchmark_config->model_id_random_seed =
        root["model_id_random_seed"].asUInt();
      if (benchmark_config->model_id_random_seed == 0) {
        TFLITE_LOG(WARN) << "Because `model_id_random_seed` == 0, the request "
                         << "generator thread will ignore the seed and use "
                         << "current timestamp as seed instead.";
      }
    }
    // Set Model Configurations
    for (int i = 0; i < root["models"].size(); ++i) {
      std::vector<InputLayerInfo> input_layer_info;
      ModelConfig model;
      Json::Value model_json_value = root["models"][i];
      if (model_json_value["graph"].isNull() ||
          model_json_value["period_ms"].isNull()) {
        TFLITE_LOG(ERROR) << "Please check if arguments `graph` and `period_ms`"
                             " are given in the model configs.";
        return kTfLiteError;
      }
      model.model_fname = model_json_value["graph"].asString();
      model.period_ms = model_json_value["period_ms"].asInt();
      if (model.period_ms <= 0) {
        TFLITE_LOG(ERROR) << "Please check if `period_ms` are positive.";
        return kTfLiteError;
      }

      // Set `batch_size`.
      // If no `batch_size` is given, the default batch size will be set to 1.
      if (!model_json_value["batch_size"].isNull())
        model.batch_size = model_json_value["batch_size"].asInt();

      // Set `device`.
      // Fixes to the device if specified in case of `FixedDevicePlanner`.
      if (!model_json_value["device"].isNull())
        model.device = model_json_value["device"].asInt();

      // Bounds checking is done internally in interpreter and planner, so
      // we don't check the actual values here.
      // See struct ModelConfig for default value.
      if (!model_json_value["slo_us"].isNull()) {
        model.slo_us = model_json_value["slo_us"].asInt64();
      }

      // Bounds checking is done internally in interpreter, so
      // we don't check the actual values here.
      // See struct ModelConfig for default value.
      if (!model_json_value["slo_scale"].isNull()) {
        model.slo_scale = model_json_value["slo_scale"].asFloat();
      }

      if (!model_json_value["input_layer"].isNull() &&
          !model_json_value["input_layer_shape"].isNull()) {
        TF_LITE_ENSURE_STATUS(PopulateInputLayerInfo(
            model_json_value["input_layer"].asString(),
            model_json_value["input_layer_shape"].asString(),
            model_json_value["input_layer_value_range"].asString(),
            model_json_value["input_layer_value_files"].asString(),
            &input_layer_info));
      }

      benchmark_config->model_information.push_back({input_layer_info, model});
    }

    if (benchmark_config->model_information.size() == 0) {
      TFLITE_LOG(ERROR) << "Please specify at list one model "
                        << "in `models` argument.";
      return kTfLiteError;
    }
  }

  TFLITE_LOG(INFO) << root;

  return kTfLiteOk;
}

}  // namespace tflite
