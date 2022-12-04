#include "tensorflow/lite/splash/processor_latency_model.h"
#include <cerrno>
#include <cassert>
#include <fstream>

#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/builtin_ops.h"

#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "libtflite", __VA_ARGS__)
#include <android/log.h>
#else
#define LOGI(...) printf(__VA_ARGS__)
#endif // defined(__ANDROID__)

namespace tflite {
namespace impl {

using namespace std;

TfLiteStatus ProcessorLatencyModel::Init(ResourceConfig& config) {
  model_path_ = config.latency_model_param_path;
  LoadModelParameter(model_path_);
  return kTfLiteOk;
}

void ProcessorLatencyModel::LoadModelParameter(string latency_model_path) {
  Json::Value model_param = LoadJsonObjectFromFile(latency_model_path); 
  for (auto worker_id_it = model_param.begin(); worker_id_it != model_param.end(); ++worker_id_it) {
    int worker_id = std::atoi(worker_id_it.key().asString().c_str());
    if (worker_id != wid_) {
      continue;
    }

    const Json::Value param = *worker_id_it;
    for (auto model_it = param.begin(); model_it != param.end(); ++model_it) {
      int model_id = std::atoi(model_it.key().asString().c_str());
      model_latency_table_[model_id] = std::unordered_map<int, int64_t>(); 

      const Json::Value model = *model_it;
      for (auto temp_it = model.begin(); temp_it != model.end(); ++temp_it) {
        int temp = std::atoi(temp_it.key().asString().c_str());
        int64_t latency = (*temp_it).asInt64(); 
        
        if (latency <= 0) {
          continue;
        }

        LOGI("[ProcessorLatencyModel][model_id = %d] temp(%d) : latency = %lld", model_id, temp, latency);
        model_latency_table_[model_id][temp] = latency;
      }
    }
  }
}

int64_t ProcessorLatencyModel::Predict(Subgraph* subgraph) {
  thermal_t target_temp = 25;
  if (is_thermal_aware_) {
    target_temp = GetResourceMonitor().GetTemperature(wid_) / 1000; 
  } 
  int model_id = subgraph->GetKey().model_id;
  auto it = model_latency_table_.find(model_id);
  auto model_count = minimum_profiled_count_.find(model_id);
  if (model_count == minimum_profiled_count_.end() || model_count->second < minimum_profiled_threshold_) {
    // No model or has model but less than minimum
    return 0;
  }
  auto model_latency = model_latency_table_.find(model_id); 
  if (model_latency == model_latency_table_.end()) {
    // No model
    return 0;
  }
  auto model_latency_temp = model_latency->second.find(target_temp);
  if (model_latency_temp != model_latency->second.end()) {
    return model_latency_temp->second;
  } else {
    if (is_thermal_aware_) {
      return FindNearestValue(model_id, target_temp);
    }
    else return 0;
  }
}

int64_t ProcessorLatencyModel::FindNearestValue(int model_id, thermal_t target_temp) {
  auto model_latency = model_latency_table_.find(model_id); 
  for (thermal_t i = target_temp ; i >= 0 ; i--) {
    auto model_latency_temp = model_latency->second.find(i);
    if (model_latency_temp != model_latency->second.end() && model_latency_temp->second > 0) {
      return model_latency_temp->second;
    } 
  }
  for (thermal_t i = target_temp ; i <= 100 ; i++) {
    auto model_latency_temp = model_latency->second.find(i);
    if (model_latency_temp != model_latency->second.end() && model_latency_temp->second > 0) {
      return model_latency_temp->second;
    } 
  }
  return 0;
}

TfLiteStatus ProcessorLatencyModel::Profile(int32_t model_id, int64_t latency) {
  thermal_t target_temp = GetResourceMonitor().GetTemperature(wid_) / 1000; 
  if (model_latency_table_.find(model_id) == model_latency_table_.end()) {
    model_latency_table_[model_id] = std::unordered_map<int, int64_t>();
  }
  model_latency_table_[model_id][target_temp] = latency;
  return kTfLiteOk;
}

TfLiteStatus ProcessorLatencyModel::Update(Job job, Subgraph* subgraph) {
  thermal_t target_temp = 25;
  if (is_thermal_aware_) {
    target_temp = GetResourceMonitor().GetTemperature(wid_) / 1000; 
  }
  int model_id = job.model_id;
  auto it = model_latency_table_.find(model_id);
  if (it != model_latency_table_.end()) {
    auto latency = it->second.find(target_temp);
    auto prev_latency = 0;
    if (latency != it->second.end()) {
      prev_latency = latency->second;
    } else {
      prev_latency = FindNearestValue(model_id, target_temp);
    }
    if (prev_latency == 0) {
      model_latency_table_[model_id][target_temp] = job.latency;  
    } else {
      if (job.latency > prev_latency * 3) return kTfLiteOk; // Outlier
      model_latency_table_[model_id][target_temp] = 
          smoothing_factor_ * job.latency + (1 - smoothing_factor_) * prev_latency; 
    }
    auto count = minimum_profiled_count_[model_id];
    if (count <= minimum_profiled_threshold_) {
      minimum_profiled_count_[model_id] = count + 1;
    }
  } else {
    model_latency_table_[model_id] = std::unordered_map<int, int64_t>(); 
    model_latency_table_[model_id][target_temp] = 0;
    minimum_profiled_count_[model_id] = 1;
  }
  return kTfLiteOk;
}

TfLiteStatus ProcessorLatencyModel::Close() {
  Json::Value root;
  if (wid_ != 0) {
    root = LoadJsonObjectFromFile(model_path_); 
  }
  Json::Value comp;
  for (auto comp_time : model_latency_table_) {
    Json::Value model;
    for (auto temp_latency: comp_time.second) {
      if (temp_latency.second != 0) {
        model[std::to_string(temp_latency.first)] = temp_latency.second;
      }
    }
    comp[std::to_string(comp_time.first)] = model;
  }
  root[std::to_string(wid_)] = comp;
  WriteJsonObjectToFile(root, model_path_);
  return kTfLiteOk;
}

} // namespace impl
} // namespace tflite
