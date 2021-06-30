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

#include <string>

#include "tensorflow/lite/profiling/util.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace profiling {
namespace util {

ModelDeviceToLatency ConvertModelNameToId(const Json::Value name_profile,
                                          std::map<int, ModelConfig>& model_configs) {
  ModelDeviceToLatency id_profile;
  for (auto name_profile_it = name_profile.begin();
       name_profile_it != name_profile.end(); ++name_profile_it) {
    std::string model_name = name_profile_it.key().asString();

    // check the integer id of this model name
    int model_id = GetModelId(model_name, model_configs);
    if (model_id == -1) {
      // we're not interested in this model for this run
      continue;
    }

    const Json::Value idx_profile = *name_profile_it;
    for (auto idx_profile_it = idx_profile.begin();
         idx_profile_it != idx_profile.end(); ++idx_profile_it) {
      std::string idx = idx_profile_it.key().asString();

      // parse the key to retrieve start/end indices
      // e.g., "25/50" --> delim_pos = 2
      auto delim_pos = idx.find("/");
      std::string start_idx = idx.substr(0, delim_pos);
      std::string end_idx = idx.substr(delim_pos + 1, idx.length() - delim_pos - 1);
      
      const Json::Value device_profile = *idx_profile_it;
      for (auto device_profile_it = device_profile.begin();
           device_profile_it != device_profile.end();
           ++device_profile_it) {
        int device_id = device_profile_it.key().asInt();
        TfLiteDeviceFlags device_flag = static_cast<TfLiteDeviceFlags>(device_id);
        int64_t profiled_latency = (*device_profile_it).asInt64();

        if (profiled_latency <= 0) {
          // jsoncpp treats missing values (null) as zero,
          // so they will be filtered out here
          continue;
        }

        SubgraphKey key(model_id, device_flag,
                        std::stoi(start_idx), std::stoi(end_idx));
        id_profile[key] = profiled_latency;
      }
    }
  }
  return id_profile;
}

void ConvertModelIdToName(const ModelDeviceToLatency id_profile,
                          Json::Value& name_profile,
                          std::map<int, ModelConfig>& model_configs);
  for (auto& pair : id_profile) {
    SubgraphKey key = pair.first;
    int model_id = key.model_id;
    std::string start_idx = std::to_string(key.start_idx);
    std::string end_idx = std::to_string(key.end_idx);
    int64_t profiled_latency = pair.second;

    // check the string name of this model id
    std::string model_name = GetModelName(key.model_id, model_configs);
    if (model_name.empty()) {
      TFLITE_LOG(WARN) << "Cannot find model #" << model_id << ". Will ignore.";
      continue;
    }

    // copy all entries in id_profile --> name_profile
    // as an ad-hoc method, we simply concat the start/end indices to form
    // the level-two key in the final json value
    name_profile[model_name][start_idx + "/" + end_idx][key.device_flag] = profiled_latency;
  }
}

}  // namespace util
}  // namespace profiling
}  // namespace tflite
