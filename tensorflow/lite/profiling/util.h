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
#ifndef TENSORFLOW_LITE_PROFILING_UTIL_H_
#define TENSORFLOW_LITE_PROFILING_UTIL_H_

#include <map>
#include <json/json.h>

#include "tensorflow/lite/util.h"

namespace tflite {
namespace profiling {
namespace util {

using ModelDeviceToLatency = std::map<SubgraphKey, int64_t>;
using ModelDeviceToFrequencyLatency = std::map<SubgraphKey, std::map<int64_t, int64_t>>;

// Convert entries in the json value to ModelDeviceToLatency format,
// for the given model name and id.
// The return val can be given to the interpreter.
ModelDeviceToLatency ExtractModelProfile(const Json::Value& name_profile,
                                         const std::string& model_fname,
                                         const int model_id);
ModelDeviceToFrequencyLatency ExtractModelFrequencyProfile(
    const Json::Value& name_frequency_profile, const std::string& model_fname,
    const int model_id);

// Convert model integer ids back to string-type names for model profiles,
// and update database_json with the newly updated profiled latency values.
// This function does not erase entries in database_json for models that were
// not run during this benchmark run.
void UpdateStaticDatabase(const ModelDeviceToLatency& id_profile,
                          const std::map<int, ModelConfig>& model_configs,
                          Json::Value& database_json);
void UpdateFrequencyDatabase(
    const ModelDeviceToFrequencyLatency& id_frequency_profile,
    const std::map<int, ModelConfig>& model_configs,
    Json::Value& frequency_database_json);
}  // namespace util
}  // namespace profiling
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PROFILING_UTIL_H_
