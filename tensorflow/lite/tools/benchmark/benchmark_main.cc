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

#include "tensorflow/lite/tools/benchmark/multimodel_benchmark.h"

namespace tflite {
namespace benchmark {

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
        multimodel_benchmark.RunRequests();
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
