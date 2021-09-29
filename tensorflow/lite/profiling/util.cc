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

ModelDeviceToLatency ExtractModelProfile(const Json::Value& name_profile,
                                         const std::string& model_fname,
                                         const int model_id) {
  auto string_to_node_indices = [](std::string index_string) {
    std::set<int> node_indices;
    std::stringstream ss(index_string);

    for (int i; ss >> i;) {
      node_indices.insert(i);
      if (ss.peek() == ',') {
        ss.ignore();
      }
    }

    return node_indices;
  };
  
  ModelDeviceToLatency id_profile;
  for (auto name_profile_it = name_profile.begin();
       name_profile_it != name_profile.end(); ++name_profile_it) {
    std::string model_name = name_profile_it.key().asString();

    if (model_name != model_fname) {
      // We're only interested in `model_fname`.
      // NOTE: In case a model is using a different string name alias for
      // some other reason (e.g., two instances of the same model), we won't
      // be able to detect that the model can indeed reuse this profile.
      // An ad-hoc fix would be to add yet another "model name" field,
      // solely for profiling purposes.
      continue;
    }

    const Json::Value idx_profile = *name_profile_it;
    for (auto idx_profile_it = idx_profile.begin();
         idx_profile_it != idx_profile.end(); ++idx_profile_it) {
      std::string idx = idx_profile_it.key().asString();

      // parse the key to retrieve start/end indices
      // e.g., "25/50" --> delim_pos = 2
      auto delim_pos = idx.find("/");
      std::set<int> root_indices =
          string_to_node_indices(idx.substr(0, delim_pos));
      std::set<int> leaf_indices =
          string_to_node_indices(idx.substr(delim_pos + 1, idx.length() - delim_pos - 1));
      
      const Json::Value device_profile = *idx_profile_it;
      for (auto device_profile_it = device_profile.begin();
           device_profile_it != device_profile.end();
           ++device_profile_it) {
        int worker_id = device_profile_it.key().asInt();
        int64_t profiled_latency = (*device_profile_it).asInt64();

        if (profiled_latency <= 0) {
          // jsoncpp treats missing values (null) as zero,
          // so they will be filtered out here
          continue;
        }

        SubgraphKey key(model_id, worker_id,
                        root_indices, leaf_indices);
        id_profile[key] = profiled_latency;
      }
    }
  }
  return id_profile;
}

ModelDeviceToFrequencyLatency ExtractModelFrequencyProfile(
    const Json::Value& name_frequency_profile, const std::string& model_fname,
    const int model_id) {
  auto string_to_node_indices = [](std::string index_string) {
    std::set<int> node_indices;
    std::stringstream ss(index_string);

    for (int i; ss >> i;) {
      node_indices.insert(i);
      if (ss.peek() == ',') {
        ss.ignore();
      }
    }

    return node_indices;
  };
  
  ModelDeviceToFrequencyLatency id_frequency_profile;
  for (auto name_frequency_profile_it = name_frequency_profile.begin();
       name_frequency_profile_it != name_frequency_profile.end();
       ++name_frequency_profile_it) {
    std::string model_name = name_frequency_profile_it.key().asString();

    if (model_name != model_fname) {
      // We're only interested in `model_fname`.
      // NOTE: In case a model is using a different string name alias for
      // some other reason (e.g., two instances of the same model), we won't
      // be able to detect that the model can indeed reuse this profile.
      // An ad-hoc fix would be to add yet another "model name" field,
      // solely for profiling purposes.
      continue;
    }

    const Json::Value idx_frequency_profile = *name_frequency_profile_it;
    for (auto idx_frequency_profile_it = idx_frequency_profile.begin();
         idx_frequency_profile_it != idx_frequency_profile.end();
         ++idx_frequency_profile_it) {
      std::string idx = idx_frequency_profile_it.key().asString();

      // parse the key to retrieve start/end indices
      // e.g., "25/50" --> delim_pos = 2
      auto delim_pos = idx.find("/");
      std::set<int> root_indices =
          string_to_node_indices(idx.substr(0, delim_pos));
      std::set<int> leaf_indices =
          string_to_node_indices(idx.substr(delim_pos + 1, idx.length() - delim_pos - 1));

      const Json::Value device_frequency_profile = *idx_frequency_profile_it;
      for (auto device_frequency_profile_it = device_frequency_profile.begin();
           device_frequency_profile_it != device_frequency_profile.end();
           ++device_frequency_profile_it) {
        int worker_id = std::stoi(device_frequency_profile_it.key().asString());

        std::map<int64_t, int64_t> frequency_to_latency;
        const Json::Value frequency_profile = *device_frequency_profile_it;
        for (auto frequency_profile_it = frequency_profile.begin();
             frequency_profile_it != frequency_profile.end();
             ++frequency_profile_it) {
          int64_t frequency = std::stoi(frequency_profile_it.key().asString());
          int64_t latency = (*frequency_profile_it).asInt64();
          if (latency <= 0) {
            // jsoncpp treats missing values (null) as zero,
            // so they will be filtered out here
            continue;
          }
          frequency_to_latency[frequency] = latency;
        }
        SubgraphKey key(model_id, worker_id, root_indices, leaf_indices);
        id_frequency_profile[key] = frequency_to_latency;
      }
    }
  }
  return id_frequency_profile;
}

void UpdateStaticDatabase(const ModelDeviceToLatency& id_profile,
                          const std::map<int, ModelConfig>& model_configs,
                          Json::Value& database_json) {
  for (auto& pair : id_profile) {
    SubgraphKey key = pair.first;
    int model_id = key.model_id;
    std::string start_indices = key.GetInputOpsString();
    std::string end_indices = key.GetOutputOpsString();
    int64_t profiled_latency = pair.second;

    // check the string name of this model id
    std::string model_name = GetModelName(key.model_id, model_configs);
    if (model_name.empty()) {
      TFLITE_LOG(WARN) << "UpdateStaticDatabase: Cannot find model #"
                       << model_id << " in model_configs. Will ignore.";
      continue;
    }

    // copy all entries in id_profile --> database_json
    // as an ad-hoc method, we simply concat the start/end indices to form
    // the level-two key in the final json value
    database_json[model_name][start_indices + "/" + end_indices][key.worker_id] = profiled_latency;
  }
}

void UpdateFrequencyDatabase(
    const ModelDeviceToFrequencyLatency& id_frequency_profile,
    const std::map<int, ModelConfig>& model_configs,
    Json::Value& frequency_database_json) {
  for (auto& pair : id_frequency_profile) {
    SubgraphKey key = pair.first;
    int model_id = key.model_id;
    std::string start_indices = key.GetInputOpsString();
    std::string end_indices = key.GetOutputOpsString();
    const std::map<int64_t, int64_t>& frequency_profiled_latency = pair.second;

    // check the string name of this model id
    std::string model_name = GetModelName(key.model_id, model_configs);
    if (model_name.empty()) {
      TFLITE_LOG(WARN) << "UpdateFrequencyDatabase: Cannot find model #"
                       << model_id << " in model_configs. Will ignore.";
      continue;
    }

    // copy all entries in id_frequency_profile --> frequency_database_json
    // as an ad-hoc method, we simply concat the start/end indices to form
    // the level-two key in the final json value
    for (const auto frequency_to_latency : frequency_profiled_latency) {
      const int64_t& frequency = frequency_to_latency.first;
      const int64_t& latency = frequency_to_latency.second;
      frequency_database_json[model_name][start_indices + "/" + end_indices]
                             [std::to_string(key.worker_id)]
                             [std::to_string(frequency)] = latency;
    }
  }
}

}  // namespace util
}  // namespace profiling
}  // namespace tflite
