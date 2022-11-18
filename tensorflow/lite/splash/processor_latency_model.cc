#include "tensorflow/lite/splash/processor_latency_model.h"
#include <cerrno>
#include <cassert>
#include <fstream>

#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/builtin_ops.h"

#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "libtflite", __VA_ARGS__)
#include <android/log.h>
#endif // defined(__ANDROID__)

namespace tflite {
namespace impl {

using namespace std;

TfLiteStatus ProcessorLatencyModel::Init() {
  // Do nothing
  return kTfLiteOk;
}

int64_t ProcessorLatencyModel::Predict(Subgraph* subgraph) {
  auto it = model_latency_table_.find(subgraph->GetKey().model_id);
  auto count = minimum_profiled_count_.find(subgraph->GetKey().model_id);
  if (count == minimum_profiled_count_.end() || count->second < minimum_profiled_threshold_) {
    return 0;
  }
  if (it != model_latency_table_.end()) {
    return it->second;
  } else {
    return 0; // Minimum value to be selected
  }
}

TfLiteStatus ProcessorLatencyModel::Profile(int32_t model_id, int64_t latency) {
  model_latency_table_[model_id] = latency; 
  return kTfLiteOk;
}

TfLiteStatus ProcessorLatencyModel::Update(Job job, Subgraph* subgraph) {
  thermal_t current_temp = GetResourceMonitor().GetTemperature(wid_);
  thermal_t threshold = GetResourceMonitor().GetThrottlingThreshold(wid_);
  if (current_temp > threshold) {
    UpdateThrottledLatency(job.model_id, job.latency); 
    return kTfLiteOk;
  }

  auto it = model_latency_table_.find(job.model_id);
  auto count = minimum_profiled_count_.find(job.model_id);
  if (it != model_latency_table_.end()) {
    // if (IsThrottled(model_id, latency, current_temp)) { // If new throttling threshold detected
    //   LOGI("PLM::Update Newly Throttling detected in worker[%d] on current temp = %d", wid_, current_temp);
    //   GetResourceMonitor().SetThrottlingThreshold(wid_, current_temp);
    //   UpdateThrottledLatency(model_id, latency);
    // } else {
      int64_t prev_latency = model_latency_table_[job.model_id];
      model_latency_table_[job.model_id] =
          smoothing_factor_ * job.latency +
          (1 - smoothing_factor_) * prev_latency;
      if (count->second <= minimum_profiled_threshold_) {
        minimum_profiled_count_[job.model_id] = count->second + 1;
      }
    // }
  } else {
    model_latency_table_[job.model_id] = job.latency;
    minimum_profiled_count_[job.model_id] = 1;
  }
  return kTfLiteOk;
}

bool ProcessorLatencyModel::IsThrottled(int32_t model_id, int64_t latency, thermal_t current_temp) {
  auto it = model_throttled_latency_table_.find(model_id);
  if (it != model_throttled_latency_table_.end()) { 
    // If table has previous throttled_latency value,
    // the system must have experienced this temp before.
    return false;
  }
  int64_t prev_latency = model_latency_table_[model_id];
  int64_t diff = latency - prev_latency;
  int64_t target = (int64_t) (prev_latency * throttled_diff_rate_);
  if (diff > target) {
    // LOGI("PLM::Newly Throttling detected (latency) = %lld", latency);
    // LOGI("PLM::Newly Throttling detected (prev_latency) = %lld", prev_latency);
    // LOGI("PLM::Newly Throttling detected (diff) = %lld", diff);
    // LOGI("PLM::Newly Throttling detected (target) = %lld", target);
    if (current_temp > throttled_temp_min_) {
      throttle_count_++;
      if (throttle_count_ > throttle_count_threshold_) {
        LOGI("PLM::Newly Throttling detected current_temp = %d", current_temp);
        return true;
      }
    }
  }
  throttle_count_ = 0;
  return false;
}

TfLiteStatus ProcessorLatencyModel::UpdateThrottledLatency(int32_t model_id, int64_t latency) {
  auto it = model_throttled_latency_table_.find(model_id);
  if (it != model_throttled_latency_table_.end()) { 
    int64_t prev_latency = model_throttled_latency_table_[model_id];
    model_throttled_latency_table_[model_id] =
        smoothing_factor_ * latency +
        (1 - smoothing_factor_) * prev_latency;
  } else {
    model_throttled_latency_table_[model_id] = latency;
  }
  return kTfLiteOk;
}

} // namespace impl
} // namespace tflite