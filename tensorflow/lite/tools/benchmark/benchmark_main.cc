/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <thread>
#include <string>
#include <vector>
#include <fstream>
#include <json/json.h>
#include <condition_variable>
#include <mutex>
#include <deque>
#include <iostream>
#include <cstdlib>

#include "tensorflow/lite/tools/benchmark/multimodel_benchmark.h"
#include "tensorflow/lite/tools/benchmark/benchmark_tflite_model.h"
#include "tensorflow/lite/tools/logging.h"
#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace benchmark {

TfLiteStatus ParseJsonFile(std::string json_path, RuntimeConfig& runtime_config) {
  std::ifstream config(json_path, std::ifstream::binary);
  Json::Value root;
  config >> root;
  std::cout << "Read JSON Config " << std::endl;
  
  if (!root.isObject()) {
    TFLITE_LOG(ERROR) << "Please validate the json config file.";
    return kTfLiteError;
  }

  // Set Runtime Configurations
  // Optional
  if (!root["running_time_ms"].isNull())
    runtime_config.run_duration = root["running_time_ms"].asInt();
  if (!root["model_profile"].isNull()) {
    runtime_config.model_profile = root["model_profile"].asString();
    std::ifstream profile_config(runtime_config.model_profile, std::ifstream::binary);
    profile_config >> runtime_config.profile_result;
  }

  // Required
  if (root["period_ms"].isNull() ||
      root["log_path"].isNull() ||
      root["models"].isNull()) {
    TFLITE_LOG(ERROR) << "Please check if arguments "
                      << "`period_ms`, `log_path` and `models`"
                      << " are given in the config file.";
    return kTfLiteError;
  }

  runtime_config.period_ms = root["period_ms"].asInt();
  runtime_config.log_path = root["log_path"].asString();

  // Set Model Configurations
  for (int i = 0; i < root["models"].size(); ++i) {
    ModelConfig model;
    Json::Value model_json_value = root["models"][i];
    if (model_json_value["graph"].isNull()) {
      TFLITE_LOG(ERROR) << "Please check if argument `graph` is not given in "
                           "the model configs.";
      return kTfLiteError;
    }
    model.model_fname = model_json_value["graph"].asString();

    // Set `batch_size`.
    // If no `batch_size` is given, the default batch size will be set to 1.
    if (!model_json_value["batch_size"].isNull())
      model.batch_size = model_json_value["batch_size"].asInt();

    // Set `device`.
    if (!model_json_value["device"].isNull())
      model.device = model_json_value["device"].asInt();

    runtime_config.model_configs.push_back(model);
  }

  if (runtime_config.model_configs.size() == 0) {
    TFLITE_LOG(ERROR) << "Please specify at list one model "
                      << "in `models` argument.";
    return kTfLiteError;
  }
  runtime_config.num_models = runtime_config.model_configs.size();

  TFLITE_LOG(INFO) << root;

  return kTfLiteOk;
}

int Main(int argc, char** argv) {
  TFLITE_LOG(INFO) << "STARTING!!";
  BenchmarkTfLiteModel parser;
  TF_LITE_ENSURE_STATUS(parser.ParseFlags(argc, argv));
  std::string json_path = parser.params_.Get<std::string>("json_path");
  RuntimeConfig runtime_config;
  TF_LITE_ENSURE_STATUS(ParseJsonFile(json_path, runtime_config));
  
  std::vector<int> range;
  for (int i = 0; i < runtime_config.num_models; ++i) {
    range.push_back(i);
  }

  bool first = true;
  bool fail = false;
  std::vector<std::string> executed_plans;

	std::srand(5323);
	do {
    /*
    if (!first && !fail) {
      std::this_thread::sleep_for(std::chrono::milliseconds(run_duration));
    } else {
      first = false;
    }*/

		std::vector<int> device_plan;
    std::string current_plan = "";
		for (auto it = range.begin(); it != range.end(); ++it) {
			device_plan.push_back(*it);

      if (*it == 0)
        current_plan += "0";
      if (*it == 1)
        current_plan += "1";
      if (*it == 2)
        current_plan += "2";
      if (*it == 3)
        current_plan += "3";
      /*
      if (*it == 0 || *it == 1)
        current_plan += "0";
      else
        current_plan += "1";
      */
    }

    if (current_plan == "0123") {
      MultimodelBenchmark multimodel_benchmark(runtime_config, device_plan);
      TfLiteStatus status = multimodel_benchmark.Initialize(argc, argv);
      if (status == kTfLiteOk) {
        multimodel_benchmark.RunRequests(runtime_config.period_ms);
        fail = false;
      }
      else {
        fail = true;
      }
    }
    /*
    if (std::find(executed_plans.begin(), executed_plans.end(), current_plan) == executed_plans.end()) {
      executed_plans.push_back(current_plan);

      MultiModelBenchmark multimodel_benchmark(device_plan);
      TfLiteStatus status = multimodel_benchmark.Initialize(json_path, profile_data, argc, argv);
      if (status == kTfLiteOk) {
        multimodel_benchmark.RunRequests(period);
        fail = false;
      }
      else {
        fail = true;
      }
    }*/
	} while (std::next_permutation(range.begin(), range.end()));

  return EXIT_SUCCESS;
}
}  // namespace benchmark
}  // namespace tflite

int main(int argc, char** argv) { return tflite::benchmark::Main(argc, argv); }
