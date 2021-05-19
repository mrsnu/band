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

#include "absl/strings/numbers.h"
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

std::vector<std::string> Split(const std::string& str, const char delim) {
  std::vector<std::string> results;
  if (!util::SplitAndParse(str, delim, &results)) {
    results.clear();
  }
  return results;
}

int FindLayerInfoIndex(std::vector<util::InputLayerInfo>* info,
                       const std::string& input_name,
                       const std::string& names_string) {
  for (int i = 0; i < info->size(); ++i) {
    if (info->at(i).name == input_name) {
      return i;
    }
  }
  TFLITE_LOG(FATAL) << "Cannot find the corresponding input_layer name("
                    << input_name << ") in --input_layer as " << names_string;
  return -1;
}

TfLiteStatus PopulateInputValueRanges(
    const std::string& names_string, const std::string& value_ranges_string,
    std::vector<InputLayerInfo>* info) {
  std::vector<std::string> value_ranges = Split(value_ranges_string, ':');
  for (const auto& val : value_ranges) {
    std::vector<std::string> name_range = Split(val, ',');
    if (name_range.size() != 3) {
      TFLITE_LOG(ERROR) << "Wrong input value range item specified: " << val;
      return kTfLiteError;
    }

    // Ensure the specific input layer name exists.
    int layer_info_idx = FindLayerInfoIndex(info, name_range[0], names_string);

    // Parse the range value.
    int low, high;
    bool has_low = absl::SimpleAtoi(name_range[1], &low);
    bool has_high = absl::SimpleAtoi(name_range[2], &high);
    if (!has_low || !has_high || low > high) {
      TFLITE_LOG(ERROR)
          << "Wrong low and high value of the input value range specified: "
          << val;
      return kTfLiteError;
    }
    info->at(layer_info_idx).has_value_range = true;
    info->at(layer_info_idx).low = low;
    info->at(layer_info_idx).high = high;
  }
  return kTfLiteOk;
}

TfLiteStatus PopulateInputValueFiles(
    const std::string& names_string, const std::string& value_files_string,
    std::vector<InputLayerInfo>* info) {
  std::vector<std::string> value_files = Split(value_files_string, ',');
  for (const auto& val : value_files) {
    std::vector<std::string> name_file = Split(val, ':');
    if (name_file.size() != 2) {
      TFLITE_LOG(ERROR) << "Wrong input value file item specified: " << val;
      return kTfLiteError;
    }

    // Ensure the specific input layer name exists.
    int layer_info_idx = FindLayerInfoIndex(info, name_file[0], names_string);
    if (info->at(layer_info_idx).has_value_range) {
      TFLITE_LOG(WARN)
          << "The input_name:" << info->at(layer_info_idx).name
          << " appears both in input_layer_value_files and "
             "input_layer_value_range. The input_layer_value_range of the "
             "input_name will be ignored.";
    }
    info->at(layer_info_idx).input_file_path = name_file[1];
  }
  return kTfLiteOk;
}

TfLiteStatus PopulateInputLayerInfo(
    const std::string& names_string, const std::string& shapes_string,
    const std::string& value_ranges_string,
    const std::string& value_files_string,
    std::vector<InputLayerInfo>* info) {
  info->clear();
  std::vector<std::string> names = Split(names_string, ',');
  std::vector<std::string> shapes = Split(shapes_string, ':');

  if (names.size() != shapes.size()) {
    TFLITE_LOG(ERROR) << "The number of items in"
                      << " --input_layer_shape (" << shapes_string << ", with "
                      << shapes.size() << " items)"
                      << " must match the number of items in"
                      << " --input_layer (" << names_string << ", with "
                      << names.size() << " items)."
                      << " For example --input_layer=input1,input2"
                      << " --input_layer_shape=1,224,224,4:1,20";
    return kTfLiteError;
  }

  for (int i = 0; i < names.size(); ++i) {
    info->push_back(InputLayerInfo());
    InputLayerInfo& input = info->back();

    input.name = names[i];

    TFLITE_TOOLS_CHECK(SplitAndParse(shapes[i], ',', &input.shape))
        << "Incorrect size string specified: " << shapes[i];
    for (int dim : input.shape) {
      if (dim == -1) {
        TFLITE_LOG(ERROR)
            << "Any unknown sizes in the shapes (-1's) must be replaced"
            << " with the size you want to benchmark with.";
        return kTfLiteError;
      }
    }
  }

  // Populate input value range if it's specified.
  TF_LITE_ENSURE_STATUS(
      PopulateInputValueRanges(names_string, value_ranges_string, info));

  // Populate input value files if it's specified.
  TF_LITE_ENSURE_STATUS(
      PopulateInputValueFiles(names_string, value_files_string, info));

  return kTfLiteOk;
}

TfLiteStatus ParseJsonFile(std::string json_fname,
                           RuntimeConfig* runtime_config) {
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
    runtime_config->cpu_masks =
        impl::TfLiteCPUMaskGetMask(root["cpu_masks"].asCString());
  }
  if (!root["worker_cpu_masks"].isNull()) {
    for (auto const& key : root["worker_cpu_masks"].getMemberNames()) {
      size_t device_id = TfLiteDeviceGetFlag(key.c_str());
      impl::TfLiteCPUMaskFlags flag =  
          impl::TfLiteCPUMaskGetMask(root["worker_cpu_masks"][key].asCString());
      if (device_id < kTfLiteNumDevices && flag != impl::kTfLiteAll) {
        runtime_config->worker_cpu_masks[device_id] = flag;
      }
    }
  }
  if (!root["running_time_ms"].isNull())
    runtime_config->running_time_ms = root["running_time_ms"].asInt();
  if (!root["profile_smoothing_factor"].isNull())
    runtime_config->profile_smoothing_factor = root["profile_smoothing_factor"].asFloat();
  if (!root["model_profile"].isNull())
    runtime_config->model_profile = root["model_profile"].asString();
  if (!root["allow_work_steal"].isNull())
    runtime_config->allow_work_steal = root["allow_work_steal"].asBool();
  if (!root["schedule_window_size"].isNull()) {
    runtime_config->schedule_window_size = root["schedule_window_size"].asInt();
    if (runtime_config->schedule_window_size <= 0) {
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

  runtime_config->log_path = root["log_path"].asString();
  runtime_config->execution_mode = root["execution_mode"].asString();

  int planner_id = root["planner"].asInt();
  if (planner_id < kFixedDevice || planner_id >= kNumPlannerTypes) {
    TFLITE_LOG(ERROR) << "Wrong `planner` argument is given.";
    return kTfLiteError;
  }
  runtime_config->planner_type = static_cast<TfLitePlannerType>(planner_id);

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
      TFLITE_LOG(ERROR) << "Please check if arguments `period_ms` are positive.";
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

    if (!model_json_value["input_layer"].isNull() &&
        !model_json_value["input_layer_shape"].isNull()) {
      TF_LITE_ENSURE_STATUS(PopulateInputLayerInfo(
          model_json_value["input_layer"].asString(),
          model_json_value["input_layer_shape"].asString(),
          model_json_value["input_layer_value_range"].asString(),
          model_json_value["input_layer_value_files"].asString(),
          &input_layer_info));
    }

    runtime_config->model_information.push_back({input_layer_info, model});
  }

  if (runtime_config->model_information.size() == 0) {
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
