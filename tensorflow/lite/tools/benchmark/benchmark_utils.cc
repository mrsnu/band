/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <json/json.h>
#include <fstream>

#include "tensorflow/lite/tools/benchmark/benchmark_utils.h"
#include "tensorflow/lite/tools/logging.h"
#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace benchmark {
namespace util {

void SleepForSeconds(double sleep_seconds) {
  if (sleep_seconds <= 0.0) {
    return;
  }
  // If requested, sleep between runs for an arbitrary amount of time.
  // This can be helpful to determine the effect of mobile processor
  // scaling and thermal throttling.
  tflite::profiling::time::SleepForMicros(
      static_cast<uint64_t>(sleep_seconds * 1e6));
}

TfLiteStatus LoadGen::ParseJsonFile(std::string json_fname) {
  std::ifstream config(json_fname, std::ifstream::binary);
  Json::Value root;
  config >> root;

  if (!root.isObject()) {
    TFLITE_LOG(ERROR) << "Please validate the json config file.";
    return kTfLiteError;
  }

  // Note : program aborts when asX fails below
  // e.g., asInt, asCString, ...

  // Set Runtime Configurations
  // Optional
  if (!root["cpu_masks"].isNull()) {
    runtime_config_.cpu_masks =
        impl::TfLiteCPUMaskGetMask(root["cpu_masks"].asCString());
  }
  if (!root["worker_cpu_masks"].isNull()) {
    for (auto const& key : root["worker_cpu_masks"].getMemberNames()) {
      size_t device_id = TfLiteDeviceGetFlag(key.c_str());
      impl::TfLiteCPUMaskFlags flag =
          impl::TfLiteCPUMaskGetMask(root["worker_cpu_masks"][key].asCString());
      if (device_id < kTfLiteNumDevices && flag != impl::kTfLiteAll) {
        runtime_config_.worker_cpu_masks[device_id] = flag;
      }
    }
  }
  if (!root["running_time_ms"].isNull()) {
    runtime_config_.running_time_ms = root["running_time_ms"].asInt();
  }
  if (!root["profile_smoothing_factor"].isNull()) {
    runtime_config_.profile_smoothing_factor =
      root["profile_smoothing_factor"].asFloat();
  }
  if (!root["model_profile"].isNull()) {
    runtime_config_.model_profile = root["model_profile"].asString();
  }
  if (!root["allow_work_steal"].isNull()) {
    runtime_config_.allow_work_steal = root["allow_work_steal"].asBool();
  }
  if (!root["schedule_window_size"].isNull()) {
    runtime_config_.schedule_window_size = root["schedule_window_size"].asInt();
    if (runtime_config_.schedule_window_size <= 0) {
      TFLITE_LOG(ERROR) << "Make sure `schedule_window_size` > 0.";
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

  runtime_config_.log_path = root["log_path"].asString();
  runtime_config_.execution_mode = root["execution_mode"].asString();

  int planner_id = root["planner"].asInt();
  if (planner_id < kFixedDevice || planner_id >= kNumPlannerTypes) {
    TFLITE_LOG(ERROR) << "Wrong `planner` argument is given.";
    return kTfLiteError;
  }
  runtime_config_.planner_type = static_cast<TfLitePlannerType>(planner_id);

  // Set Model Configurations
  for (int i = 0; i < root["models"].size(); ++i) {
    ModelConfig model;
    Json::Value model_json_value = root["models"][i];
    if (model_json_value["graph"].isNull() ||
        model_json_value["period_ms"].isNull()) {
      TFLITE_LOG(ERROR) << "Please check if arguments `graph` and `period_ms`"
                           " are given in the model configs.";
      return kTfLiteError;
    }
    model.model_id = runtime_config_.model_configs.size();
    model.model_fname = model_json_value["graph"].asString();
    model.period_ms = model_json_value["period_ms"].asInt();
    if (model.period_ms <= 0) {
      TFLITE_LOG(ERROR) << "Please check if `period_ms` is positive.";
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

    runtime_config_.model_configs.push_back(model);
  }

  if (runtime_config_.model_configs.size() == 0) {
    TFLITE_LOG(ERROR) << "Please specify at list one model "
                      << "in `models` argument.";
    return kTfLiteError;
  }

  TFLITE_LOG(INFO) << root;

  return kTfLiteOk;
}
}  // namespace util
}  // namespace benchmark
}  // namespace tflite
