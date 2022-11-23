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

TfLiteStatus ProcessorLatencyModel::Init() {
  // Do nothing
  return kTfLiteOk;
}

int64_t ProcessorLatencyModel::Predict(Subgraph* subgraph) {
  thermal_t target_temp = GetResourceMonitor().GetTargetTemperature(wid_) / 1000; 
  int model_id = subgraph->GetKey().model_id;
  auto it = model_latency_table_.find(model_id);
  auto model_count = minimum_profiled_count_.find(model_id);
  if (model_count == minimum_profiled_count_.end() || model_count->second < minimum_profiled_threshold_) {
    return 0;
  }
  auto model_latency = model_latency_table_.find(model_id); 
  if (model_latency == model_latency_table_.end()) {
    return 0; // Minimum value to be selected
  }
  auto model_latency_temp = model_latency->second.find(target_temp);
  if (model_latency_temp != model_latency->second.end()) {
    return model_latency_temp->second;
  } else {
    return FindNearestValue(model_id, target_temp);
  }
}

int64_t ProcessorLatencyModel::FindNearestValue(int model_id, thermal_t target_temp) {
  auto model_latency = model_latency_table_.find(model_id); 
  for (thermal_t i = target_temp ; i >= 0 ; i--) {
    auto model_latency_temp = model_latency->second.find(i);
    if (model_latency_temp != model_latency->second.end()) {
      return model_latency_temp->second;
    } 
  }
}

TfLiteStatus ProcessorLatencyModel::Profile(int32_t model_id, int64_t latency) {
  thermal_t target_temp = GetResourceMonitor().GetTargetTemperature(wid_) / 1000; 
  if (model_latency_table_.find(model_id) == model_latency_table_.end()) {
    model_latency_table_[model_id] = std::unordered_map<int, int64_t>();
  }
  model_latency_table_[model_id][target_temp] = latency;
  return kTfLiteOk;
}

TfLiteStatus ProcessorLatencyModel::Update(Job job, Subgraph* subgraph) {
  thermal_t target_temp = GetResourceMonitor().GetTargetTemperature(wid_) / 1000; 
  int model_id = job.model_id;
  auto it = model_latency_table_.find(model_id);
  if (it != model_latency_table_.end()) {
    auto latency = it->second.find(target_temp);
    if (latency != it->second.end()) {
      int64_t prev_latency = latency->second;
      model_latency_table_[model_id][target_temp] =
          smoothing_factor_ * job.latency +
          (1 - smoothing_factor_) * prev_latency;
      auto count = minimum_profiled_count_[model_id];
      if (count <= minimum_profiled_threshold_) {
        minimum_profiled_count_[model_id] = count + 1;
      }
    } else {
      model_latency_table_[model_id][target_temp] = 0; 
    }
  } else {
    model_latency_table_[model_id] = std::unordered_map<int, int64_t>(); 
    model_latency_table_[model_id][target_temp] = 0;
    minimum_profiled_count_[model_id] = 1;
  }
  return kTfLiteOk;
}

} // namespace impl
} // namespace tflite