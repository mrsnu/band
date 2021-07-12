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

#ifndef TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_UTILS_H_
#define TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_UTILS_H_

#include <sstream>
#include <string>
#include <vector>

#include "tensorflow/lite/cpu.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace benchmark {
namespace util {

struct InputLayerInfo {
  InputLayerInfo() : has_value_range(false) {}

  std::string name;
  std::vector<int> shape;

  // The input value is randomly generated when benchmarking the NN model.
  // However, the NN model might require the value be limited to a certain
  // range [low, high] for this particular input layer. For simplicity,
  // support integer value first.
  bool has_value_range;
  int low;
  int high;

  // The input value will be loaded from 'input_file_path' INSTEAD OF being
  // randomly generated. Note the input file will be opened in binary mode.
  std::string input_file_path;
};

// Implement type erasure with unique_ptr with custom deleter.
using VoidUniquePtr = std::unique_ptr<void, void (*)(void*)>;

struct InputTensorData {
  InputTensorData() : data(nullptr, nullptr) {}

  VoidUniquePtr data;
  size_t bytes;
};

struct ModelInformation {
  ModelInformation(std::vector<InputLayerInfo> input_layer_infos,
                   ModelConfig config)
    :input_layer_infos(input_layer_infos), config(config) {}
  std::vector<InputLayerInfo> input_layer_infos;
  std::vector<InputTensorData> input_tensor_data;
  ModelConfig config;
};

struct BenchmarkConfig {
  std::string execution_mode;
  unsigned model_id_random_seed;
  int global_period_ms;
  int running_time_ms = 60000;
  std::vector<ModelInformation> model_information;
};

TfLiteStatus ParseBenchmarkConfigFromJson(std::string json_fname,
                                          BenchmarkConfig& benchmark_config);

// A convenient function that wraps tflite::profiling::time::SleepForMicros and
// simply return if 'sleep_seconds' is negative.
void SleepForSeconds(double sleep_seconds);

// Split the 'str' according to 'delim', and store each splitted element into
// 'values'.
template <typename T>
bool SplitAndParse(const std::string& str, char delim, std::vector<T>* values) {
  std::istringstream input(str);
  for (std::string line; std::getline(input, line, delim);) {
    std::istringstream to_parse(line);
    T val;
    to_parse >> val;
    if (!to_parse.eof() && !to_parse.good()) {
      return false;
    }
    values->emplace_back(val);
  }
  return true;
}

}  // namespace util
}  // namespace benchmark
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_UTILS_H_
